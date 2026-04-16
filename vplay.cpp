/*
 * VPLAY — Minimal C++ Video Player
 * Dependencies : FFmpeg (libavcodec, libavformat, libavutil, libswscale,
 *                        libswresample)
 *                SDL2
 *
 * Build (Linux/macOS):
 *   g++ -std=c++17 -O2 vplay.cpp \
 *       $(pkg-config --cflags --libs sdl2 libavcodec libavformat \
 *                    libavutil libswscale libswresample) \
 *       -o vplay
 *
 * Build (Windows, MSYS2/MinGW):
 *   g++ -std=c++17 -O2 vplay.cpp \
 *       $(pkg-config --cflags --libs sdl2 libavcodec libavformat \
 *                    libavutil libswscale libswresample) \
 *       -o vplay.exe
 *
 * Usage:
 *   ./vplay <video_file>
 *
 * Controls:
 *   Space       – Play / Pause
 *   Left/Right  – Seek ±10 s
 *   Up/Down     – Volume ±5 %
 *   M           – Mute toggle
 *   Q / ESC     – Quit
 */

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <SDL3/SDL.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>
#include <mutex>
#include <thread>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
constexpr int   AUDIO_BUFFER_SIZE = 4096;
constexpr int   SAMPLE_RATE       = 44100;
constexpr int   CHANNELS          = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe packet queue
// ─────────────────────────────────────────────────────────────────────────────
struct PacketQueue {
    std::queue<AVPacket *> q;
    std::mutex             mtx;
    std::atomic<bool>      finished{false};

    void push(AVPacket *pkt) {
        std::lock_guard<std::mutex> lk(mtx);
        q.push(pkt);
    }

    bool pop(AVPacket *&pkt) {
        std::lock_guard<std::mutex> lk(mtx);
        if (q.empty()) return false;
        pkt = q.front();
        q.pop();
        return true;
    }

    void flush() {
        std::lock_guard<std::mutex> lk(mtx);
        while (!q.empty()) {
            av_packet_free(&q.front());
            q.pop();
        }
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mtx);
        return q.size();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Player state
// ─────────────────────────────────────────────────────────────────────────────
struct Player {
    // Format
    AVFormatContext *fmt_ctx   = nullptr;
    int              v_idx     = -1;
    int              a_idx     = -1;

    // Video
    AVCodecContext  *v_ctx     = nullptr;
    SwsContext      *sws       = nullptr;
    SDL_Texture     *texture   = nullptr;

    // Audio
    AVCodecContext  *a_ctx     = nullptr;
    SwrContext      *swr       = nullptr;
    SDL_AudioDeviceID audio_dev = 0;

    // Queues
    PacketQueue      v_queue;
    PacketQueue      a_queue;

    // Playback state
    std::atomic<bool> running  {true};
    std::atomic<bool> paused   {false};
    std::atomic<float> volume  {1.0f};
    std::atomic<bool> muted    {false};
    std::atomic<double> seek_to{-1.0};  // seconds, -1 = no pending seek

    // Timing
    double           pts_base  = 0.0;  // stream start pts in seconds
    Uint32           wall_base = 0;    // SDL_GetTicks at pts_base

    // SDL
    SDL_Window      *window    = nullptr;
    SDL_Renderer    *renderer  = nullptr;
    int              win_w = 1280, win_h = 720;

    // Duration (seconds)
    double           duration  = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Audio callback (SDL)
// ─────────────────────────────────────────────────────────────────────────────
static uint8_t  s_audio_buf[AUDIO_BUFFER_SIZE * 8];
static int      s_audio_buf_size = 0;
static int      s_audio_buf_pos  = 0;
static Player  *g_player = nullptr;

static int decode_audio_frame(Player *p, uint8_t *out_buf, int out_size) {
    AVPacket *pkt = nullptr;
    AVFrame  *frame = av_frame_alloc();
    int       written = 0;

    while (written < out_size) {
        // Grab a packet
        if (!p->a_queue.pop(pkt)) {
            if (p->a_queue.finished) break;
            SDL_Delay(1);
            continue;
        }

        if (avcodec_send_packet(p->a_ctx, pkt) < 0) {
            av_packet_free(&pkt);
            continue;
        }
        av_packet_free(&pkt);

        while (avcodec_receive_frame(p->a_ctx, frame) == 0) {
            int64_t out_samples = av_rescale_rnd(
                swr_get_delay(p->swr, p->a_ctx->sample_rate) + frame->nb_samples,
                SAMPLE_RATE, p->a_ctx->sample_rate, AV_ROUND_UP);

            uint8_t *out_ptr = out_buf + written;
            int bytes = (int)out_samples * CHANNELS * 2; // S16

            if (written + bytes > out_size) {
                av_frame_unref(frame);
                av_frame_free(&frame);
                return written;
            }

            uint8_t *dst[1] = {out_ptr};
            int conv = swr_convert(p->swr, dst, (int)out_samples,
                                   (const uint8_t **)frame->data,
                                   frame->nb_samples);
            if (conv > 0)
                written += conv * CHANNELS * 2;

            av_frame_unref(frame);
        }
    }
    av_frame_free(&frame);
    return written;
}

static void audio_callback(void *userdata, uint8_t *stream, int len) {
    Player *p = static_cast<Player *>(userdata);
    SDL_memset(stream, 0, len);

    while (len > 0) {
        if (s_audio_buf_pos >= s_audio_buf_size) {
            int n = decode_audio_frame(p, s_audio_buf, sizeof(s_audio_buf));
            if (n <= 0) return;
            s_audio_buf_size = n;
            s_audio_buf_pos  = 0;
        }

        int remaining = s_audio_buf_size - s_audio_buf_pos;
        int copy      = std::min(remaining, len);

        float vol = p->muted ? 0.f : p->volume.load();
        SDL_MixAudioFormat(stream, s_audio_buf + s_audio_buf_pos,
                           AUDIO_FORMAT, copy,
                           static_cast<int>(vol * SDL_MIX_MAXVOLUME));
        // Note: SDL_MixAudioFormat requires format constant; using SDL_AUDIO_S16SYS
        s_audio_buf_pos += copy;
        stream          += copy;
        len             -= copy;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Demux thread: reads packets and routes them
// ─────────────────────────────────────────────────────────────────────────────
static void demux_thread(Player *p) {
    AVPacket *pkt = av_packet_alloc();

    while (p->running) {
        // Handle seek
        double seek_target = p->seek_to.exchange(-1.0);
        if (seek_target >= 0.0) {
            int64_t ts = static_cast<int64_t>(seek_target * AV_TIME_BASE);
            av_seek_frame(p->fmt_ctx, -1, ts, AVSEEK_FLAG_BACKWARD);
            p->v_queue.flush();
            p->a_queue.flush();
            if (p->v_ctx) avcodec_flush_buffers(p->v_ctx);
            if (p->a_ctx) avcodec_flush_buffers(p->a_ctx);
            // Reset clock
            p->pts_base  = seek_target;
            p->wall_base = SDL_GetTicks();
        }

        // Throttle if queues are large
        if (p->v_queue.size() > 200 || p->a_queue.size() > 400) {
            SDL_Delay(5);
            continue;
        }

        int ret = av_read_frame(p->fmt_ctx, pkt);
        if (ret < 0) {
            p->v_queue.finished = true;
            p->a_queue.finished = true;
            break;
        }

        if      (pkt->stream_index == p->v_idx) p->v_queue.push(av_packet_clone(pkt));
        else if (pkt->stream_index == p->a_idx) p->a_queue.push(av_packet_clone(pkt));

        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::string fmt_time(double seconds) {
    int h   = static_cast<int>(seconds) / 3600;
    int m   = (static_cast<int>(seconds) % 3600) / 60;
    int s   = static_cast<int>(seconds) % 60;
    char buf[32];
    snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, s);
    return buf;
}

static void set_window_title(Player *p, const char *filename, double pos) {
    char title[512];
    snprintf(title, sizeof(title),
             "VPLAY  |  %s  |  %s / %s",
             filename,
             fmt_time(pos).c_str(),
             fmt_time(p->duration).c_str());
    SDL_SetWindowTitle(p->window, title);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>\n";
        return 1;
    }
    const char *filename = argv[1];

    // ── FFmpeg init ───────────────────────────────────────────────────────
    Player player;
    g_player = &player;

    if (avformat_open_input(&player.fmt_ctx, filename, nullptr, nullptr) < 0) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return 1;
    }
    if (avformat_find_stream_info(player.fmt_ctx, nullptr) < 0) {
        std::cerr << "Cannot read stream info\n";
        return 1;
    }

    player.duration = player.fmt_ctx->duration / (double)AV_TIME_BASE;

    // Find video stream
    player.v_idx = av_find_best_stream(player.fmt_ctx, AVMEDIA_TYPE_VIDEO,
                                       -1, -1, nullptr, 0);
    player.a_idx = av_find_best_stream(player.fmt_ctx, AVMEDIA_TYPE_AUDIO,
                                       -1, -1, nullptr, 0);

    auto open_codec = [](AVFormatContext *fmt, int idx) -> AVCodecContext * {
        if (idx < 0) return nullptr;
        const AVCodec *codec = avcodec_find_decoder(
            fmt->streams[idx]->codecpar->codec_id);
        if (!codec) return nullptr;
        AVCodecContext *ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(ctx, fmt->streams[idx]->codecpar);
        avcodec_open2(ctx, codec, nullptr);
        return ctx;
    };

    player.v_ctx = open_codec(player.fmt_ctx, player.v_idx);
    player.a_ctx = open_codec(player.fmt_ctx, player.a_idx);

    // ── SDL init ──────────────────────────────────────────────────────────
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
        std::cerr << "SDL_Init error: " << SDL_GetError() << "\n";
        return 1;
    }

    if (player.v_ctx) {
        player.win_w = player.v_ctx->width;
        player.win_h = player.v_ctx->height;
    }

    player.window = SDL_CreateWindow(
        "VPLAY",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        player.win_w, player.win_h,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    player.renderer = SDL_CreateRenderer(player.window, -1, SDL_RENDERER_ACCELERATED);

    if (player.v_ctx) {
        player.texture = SDL_CreateTexture(
            player.renderer,
            SDL_PIXELFORMAT_YV12,
            SDL_TEXTUREACCESS_STREAMING,
            player.v_ctx->width, player.v_ctx->height);

        player.sws = sws_getContext(
            player.v_ctx->width, player.v_ctx->height, player.v_ctx->pix_fmt,
            player.v_ctx->width, player.v_ctx->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
    }

    // Audio device
    if (player.a_ctx) {
        // SwrContext
        player.swr = swr_alloc();
        av_opt_set_int       (player.swr, "in_channel_layout",
                              player.a_ctx->channel_layout, 0);
        av_opt_set_int       (player.swr, "in_sample_rate",
                              player.a_ctx->sample_rate, 0);
        av_opt_set_sample_fmt(player.swr, "in_sample_fmt",
                              player.a_ctx->sample_fmt, 0);
        av_opt_set_int       (player.swr, "out_channel_layout",
                              AV_CH_LAYOUT_STEREO, 0);
        av_opt_set_int       (player.swr, "out_sample_rate", SAMPLE_RATE, 0);
        av_opt_set_sample_fmt(player.swr, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
        swr_init(player.swr);

        SDL_AudioSpec want{}, have{};
        want.freq     = SAMPLE_RATE;
        want.format   = AUDIO_S16SYS;
        want.channels = CHANNELS;
        want.samples  = AUDIO_BUFFER_SIZE;
        want.callback = audio_callback;
        want.userdata = &player;
        player.audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        SDL_PauseAudioDevice(player.audio_dev, 0); // start playing
    }

    // ── Threads ───────────────────────────────────────────────────────────
    player.wall_base = SDL_GetTicks();
    std::thread demux(demux_thread, &player);

    // ── Main loop ─────────────────────────────────────────────────────────
    AVFrame *frame     = av_frame_alloc();
    AVFrame *frame_yuv = av_frame_alloc();

    int yuv_buf_size = 0;
    uint8_t *yuv_buf = nullptr;

    if (player.v_ctx) {
        yuv_buf_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P,
                           player.v_ctx->width, player.v_ctx->height, 1);
        yuv_buf = static_cast<uint8_t *>(av_malloc(yuv_buf_size));
        av_image_fill_arrays(frame_yuv->data, frame_yuv->linesize,
                             yuv_buf, AV_PIX_FMT_YUV420P,
                             player.v_ctx->width, player.v_ctx->height, 1);
    }

    SDL_Event   ev;
    bool        quit      = false;
    double      video_pts = 0.0;

    while (!quit && player.running) {
        // ── Events ────────────────────────────────────────────────────────
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { quit = true; break; }
            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                case SDLK_q:
                case SDLK_ESCAPE: quit = true; break;

                case SDLK_SPACE:
                    player.paused = !player.paused;
                    if (player.audio_dev)
                        SDL_PauseAudioDevice(player.audio_dev, player.paused ? 1 : 0);
                    if (player.paused) {
                        // freeze clock
                    } else {
                        player.wall_base = SDL_GetTicks() -
                            static_cast<Uint32>(video_pts * 1000.0);
                    }
                    break;

                case SDLK_LEFT:
                    player.seek_to = std::max(0.0, video_pts - 10.0);
                    break;
                case SDLK_RIGHT:
                    player.seek_to = std::min(player.duration, video_pts + 10.0);
                    break;

                case SDLK_UP:
                    player.volume = std::min(1.0f, player.volume.load() + 0.05f);
                    break;
                case SDLK_DOWN:
                    player.volume = std::max(0.0f, player.volume.load() - 0.05f);
                    break;

                case SDLK_m:
                    player.muted = !player.muted;
                    break;
                }
            }
        }

        if (quit) break;
        if (player.paused) { SDL_Delay(10); continue; }

        // ── Decode next video frame ───────────────────────────────────────
        if (!player.v_ctx) { SDL_Delay(10); continue; }

        AVPacket *pkt = nullptr;
        if (!player.v_queue.pop(pkt)) {
            if (player.v_queue.finished) break;
            SDL_Delay(1);
            continue;
        }

        if (avcodec_send_packet(player.v_ctx, pkt) == 0) {
            while (avcodec_receive_frame(player.v_ctx, frame) == 0) {
                // Compute PTS in seconds
                AVRational tb = player.fmt_ctx->streams[player.v_idx]->time_base;
                video_pts = frame->best_effort_timestamp * av_q2d(tb);

                // A/V sync: wait until wall clock catches up
                Uint32 now     = SDL_GetTicks();
                Uint32 target  = player.wall_base +
                                 static_cast<Uint32>(video_pts * 1000.0);
                if (now < target) SDL_Delay(target - now);

                // Convert & display
                sws_scale(player.sws,
                          frame->data, frame->linesize, 0, player.v_ctx->height,
                          frame_yuv->data, frame_yuv->linesize);

                SDL_UpdateYUVTexture(player.texture, nullptr,
                    frame_yuv->data[0], frame_yuv->linesize[0],
                    frame_yuv->data[1], frame_yuv->linesize[1],
                    frame_yuv->data[2], frame_yuv->linesize[2]);

                SDL_RenderClear(player.renderer);
                SDL_RenderCopy(player.renderer, player.texture, nullptr, nullptr);
                SDL_RenderPresent(player.renderer);

                set_window_title(&player, filename, video_pts);
                av_frame_unref(frame);
            }
        }
        av_packet_free(&pkt);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    player.running = false;
    demux.join();

    if (player.audio_dev) SDL_CloseAudioDevice(player.audio_dev);
    av_free(yuv_buf);
    av_frame_free(&frame);
    av_frame_free(&frame_yuv);
    sws_freeContext(player.sws);
    if (player.swr)   swr_free(&player.swr);
    if (player.v_ctx) avcodec_free_context(&player.v_ctx);
    if (player.a_ctx) avcodec_free_context(&player.a_ctx);
    avformat_close_input(&player.fmt_ctx);
    SDL_DestroyTexture(player.texture);
    SDL_DestroyRenderer(player.renderer);
    SDL_DestroyWindow(player.window);
    SDL_Quit();

    return 0;
}
