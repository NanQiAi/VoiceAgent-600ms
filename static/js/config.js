/**
 * 音频契约与协议约定。
 *
 * 这里的采样率/帧长必须与服务端逐字一致：不一致不会报错，只会表现为"声音变调/变速"
 * 或 VAD 时频特征偏移，非常难排查。
 */

// 上行 16000Hz 对齐服务端 config.SAMPLE_RATE；下行 24000Hz 对齐 Kokoro TTS。
export const SAMPLE_RATE_IN = 16000;
export const SAMPLE_RATE_OUT = 24000;

// 每帧 512 点 = 32ms @16kHz，等于服务端 CHUNK_SAMPLES / CHUNK_MS（worklet 攒够才发，双方帧长严格相等）
export const CAPTURE_CHUNK_SAMPLES = 512;

// 播放器起播水位：攒够 300ms 音频才开播，用一点延迟换抗网络抖动
export const PLAYER_PREBUFFER_SAMPLES = Math.floor(SAMPLE_RATE_OUT * 0.3);

// 首响延迟的三档配色阈值（毫秒）
export const LATENCY_GOOD_MS = 500;
export const LATENCY_OK_MS = 800;

/*
 * playableTurn —— 播放闸门，挡住"被打断轮次的在途音频"：
 *   -1 = 还没收到任何 audio_start（首轮之前宽限放行）
 *    N = audio_start(turn=N) 授权的可播轮次
 *    0 = 哨兵值，当前轮已被打断；在途帧被丢弃，直到下一轮 audio_start(N) 重新开闸
 */
export const TURN_GATE_OPEN_BEFORE_FIRST = -1;
export const TURN_GATE_CLOSED = 0;
