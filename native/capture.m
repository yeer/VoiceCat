#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <AudioToolbox/AudioToolbox.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreGraphics/CoreGraphics.h>
#import <AVFoundation/AVFoundation.h>
#include <libproc.h>

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * 采集 macOS 系统输出或指定输入设备，向 stdout 写 s16le PCM。
 * 约束：
 * - 不占用设备（不设置 hog mode），不修改系统默认输入/输出。
 * - 进程 tap 的 muteBehavior 固定为 CATapUnmuted，避免读 tap 时把扬声器静音。
 * - 只创建私有 tap / 私有聚合设备，退出时销毁。
 */

static volatile sig_atomic_t g_stop = 0;
static int g_probe = 0;
static double g_seconds = 0;
static AudioObjectID g_tap = 0;
static AudioObjectID g_aggregate = 0;
static AudioObjectID g_capture_device = 0;
static AudioDeviceIOProcID g_proc = NULL;
static AudioStreamBasicDescription g_fmt;
static int g_fmt_ready = 0;
static pthread_t g_writer;
static int g_writer_started = 0;
static SCStream *g_stream = nil;
static AVAudioEngine *g_engine = nil;
static id g_sck = nil;
static atomic_int g_sckit_phase = 0;
static char g_error[768];
static char g_device_label[256];
static char g_source_label[32];
static double g_sum_sq = 0;
static unsigned long long g_sample_count = 0;
static int g_peak_all = 0;
static int g_cleanup_done = 0;
static UInt32 g_wanted_channels = 0;

typedef struct {
    int16_t *data;
    size_t capacity_frames;
    int channels;
    atomic_ullong write_frame;
    atomic_ullong read_frame;
    atomic_ullong dropped_frames;
    atomic_ullong pushed_frames;
    atomic_uint buffers;
} Ring;

static Ring g_ring;
static int16_t *g_scratch = NULL;
static size_t g_scratch_frames = 0;
static int g_oversize_logged = 0;

static void caplog(const char *fmt, ...) {
    fputs("capture: ", stderr);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc(10, stderr);
    fflush(stderr);
}

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static double now_sec(void) {
    return CFAbsoluteTimeGetCurrent();
}

static NSString *fourcc_string(UInt32 code) {
    char s[5];
    s[0] = (char)((code >> 24) & 0xff);
    s[1] = (char)((code >> 16) & 0xff);
    s[2] = (char)((code >> 8) & 0xff);
    s[3] = (char)(code & 0xff);
    s[4] = 0;
    for (int i = 0; i < 4; i++) {
        if (s[i] < 32 || s[i] > 126) s[i] = '?';
    }
    return [NSString stringWithUTF8String:s];
}

static NSString *os_status_string(OSStatus st) {
    NSString *cc = fourcc_string((UInt32)st);
    if ([cc containsString:@"?"]) {
        return [NSString stringWithFormat:@"%d", (int)st];
    }
    return [NSString stringWithFormat:@"%d (%@)", (int)st, cc];
}

static void set_error(NSString *s) {
    const char *u = s.UTF8String ?: "未知错误";
    snprintf(g_error, sizeof(g_error), "%s", u);
}

static BOOL write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return NO;
        }
        if (w == 0) return NO;
        p += w;
        n -= (size_t)w;
    }
    return YES;
}

static void set_label(char *dst, size_t n, NSString *s) {
    if (n == 0) return;
    if (!s) s = @"";
    NSCharacterSet *nl = [NSCharacterSet newlineCharacterSet];
    NSString *clean = [[s componentsSeparatedByCharactersInSet:nl] componentsJoinedByString:@" "];
    snprintf(dst, n, "%s", clean.UTF8String ?: "");
}

static void ring_init(size_t frames, int channels) {
    free(g_ring.data);
    memset(&g_ring, 0, sizeof(g_ring));
    g_ring.data = calloc(frames * (size_t)channels, sizeof(int16_t));
    g_ring.capacity_frames = frames;
    g_ring.channels = channels;
    atomic_init(&g_ring.write_frame, 0);
    atomic_init(&g_ring.read_frame, 0);
    atomic_init(&g_ring.dropped_frames, 0);
    atomic_init(&g_ring.pushed_frames, 0);
    atomic_init(&g_ring.buffers, 0);
    free(g_scratch);
    g_scratch_frames = 16384;
    g_scratch = calloc(g_scratch_frames * (size_t)channels, sizeof(int16_t));
}

static void ring_free(void) {
    free(g_ring.data);
    g_ring.data = NULL;
    free(g_scratch);
    g_scratch = NULL;
    g_scratch_frames = 0;
}

static void ring_push(const int16_t *interleaved, size_t frames) {
    if (!frames || !g_ring.data) return;
    unsigned long long w = atomic_load_explicit(&g_ring.write_frame, memory_order_relaxed);
    unsigned long long r = atomic_load_explicit(&g_ring.read_frame, memory_order_acquire);
    size_t cap = g_ring.capacity_frames;
    size_t used = (size_t)(w - r);
    if (used > cap) used = cap;
    if (frames > cap - used) {
        atomic_fetch_add_explicit(&g_ring.dropped_frames, frames, memory_order_relaxed);
        return;
    }
    int ch = g_ring.channels;
    size_t idx = (size_t)(w % cap);
    size_t first = frames;
    if (idx + first > cap) first = cap - idx;
    memcpy(g_ring.data + idx * (size_t)ch, interleaved, first * (size_t)ch * sizeof(int16_t));
    if (first < frames) {
        memcpy(g_ring.data, interleaved + first * (size_t)ch, (frames - first) * (size_t)ch * sizeof(int16_t));
    }
    atomic_store_explicit(&g_ring.write_frame, w + frames, memory_order_release);
    atomic_fetch_add_explicit(&g_ring.pushed_frames, frames, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_ring.buffers, 1, memory_order_relaxed);
}

static size_t ring_pop(int16_t *dst, size_t max_frames) {
    unsigned long long r = atomic_load_explicit(&g_ring.read_frame, memory_order_relaxed);
    unsigned long long w = atomic_load_explicit(&g_ring.write_frame, memory_order_acquire);
    size_t avail = (size_t)(w - r);
    if (avail == 0 || !g_ring.data) return 0;
    if (avail > max_frames) avail = max_frames;
    size_t cap = g_ring.capacity_frames;
    int ch = g_ring.channels;
    size_t idx = (size_t)(r % cap);
    size_t first = avail;
    if (idx + first > cap) first = cap - idx;
    memcpy(dst, g_ring.data + idx * (size_t)ch, first * (size_t)ch * sizeof(int16_t));
    if (first < avail) {
        memcpy(dst + first * (size_t)ch, g_ring.data, (avail - first) * (size_t)ch * sizeof(int16_t));
    }
    atomic_store_explicit(&g_ring.read_frame, r + avail, memory_order_release);
    return avail;
}

static int16_t sample_at(const uint8_t *base, UInt32 index, BOOL isFloat, BOOL isSigned, UInt32 bits) {
    if (isFloat && bits == 32) {
        float x = 0;
        memcpy(&x, base + (size_t)index * 4, 4);
        if (x > 1.f) x = 1.f;
        if (x < -1.f) x = -1.f;
        return (int16_t)lrintf(x * 32767.f);
    }
    if (isSigned && bits == 16) {
        int16_t x = 0;
        memcpy(&x, base + (size_t)index * 2, 2);
        return x;
    }
    if (isSigned && bits == 32) {
        int32_t x = 0;
        memcpy(&x, base + (size_t)index * 4, 4);
        return (int16_t)(x >> 16);
    }
    return 0;
}

static int g_diag_logged = 0;
static float g_src_peak = 0;

static void note_raw_peak(const AudioBufferList *abl) {
    if (!abl || abl->mNumberBuffers == 0 || !abl->mBuffers[0].mData) return;
    const uint8_t *src = abl->mBuffers[0].mData;
    UInt32 nbytes = abl->mBuffers[0].mDataByteSize;
    UInt32 nfloat = nbytes / 4;
    if (nfloat > 256) nfloat = 256;
    for (UInt32 i = 0; i < nfloat; i++) {
        float x = 0;
        memcpy(&x, src + (size_t)i * 4, 4);
        float a = x < 0 ? -x : x;
        if (a > g_src_peak && a < 1000.f) g_src_peak = a;
    }
    if (g_diag_logged) return;
    g_diag_logged = 1;
    float first = 0;
    if (nbytes >= 4) memcpy(&first, src, 4);
    int16_t first16 = 0;
    if (nbytes >= 2) memcpy(&first16, src, 2);
    caplog("首个回调 buffers=%u ch=%u bytes=%u first_f32=%g first_s16=%d",
           abl->mNumberBuffers,
           abl->mBuffers[0].mNumberChannels,
           nbytes,
           first,
           (int)first16);
}

static int push_from_abl(const AudioBufferList *abl, const AudioStreamBasicDescription *fmt) {
    if (!abl || abl->mNumberBuffers == 0 || !fmt) return 0;
    note_raw_peak(abl);
    BOOL nonInterleaved = (fmt->mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
    BOOL isFloat = (fmt->mFormatFlags & kAudioFormatFlagIsFloat) != 0;
    BOOL isSigned = (fmt->mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0;
    UInt32 bits = fmt->mBitsPerChannel;
    UInt32 channels = fmt->mChannelsPerFrame;
    if (channels == 0) channels = (UInt32)g_ring.channels;
    if (channels == 0 || g_ring.channels <= 0) return 0;
    if ((int)channels != g_ring.channels) {
        if (!g_oversize_logged) {
            caplog("声道数与预期不一致（实际 %u，预期 %d），已丢弃", channels, g_ring.channels);
            g_oversize_logged = 1;
        }
        return 0;
    }
    UInt32 bytesPerSample = bits / 8;
    if (bytesPerSample == 0) return 0;
    const AudioBuffer *b0 = &abl->mBuffers[0];
    if (!b0->mData || b0->mDataByteSize == 0) return 0;
    UInt32 frames = 0;
    if (nonInterleaved) {
        frames = b0->mDataByteSize / bytesPerSample;
    } else {
        UInt32 bpf = bytesPerSample * channels;
        if (bpf == 0) return 0;
        frames = b0->mDataByteSize / bpf;
    }
    if (frames == 0) return 0;
    if (frames > g_scratch_frames || !g_scratch) {
        if (!g_oversize_logged) {
            caplog("音频缓冲过大（%u 帧），已丢弃", frames);
            g_oversize_logged = 1;
        }
        return 0;
    }
    int16_t *dst = g_scratch;
    if (!nonInterleaved) {
        const uint8_t *src = b0->mData;
        for (UInt32 i = 0; i < frames; i++) {
            for (UInt32 c = 0; c < channels; c++) {
                dst[i * channels + c] = sample_at(src, i * channels + c, isFloat, isSigned, bits);
            }
        }
    } else {
        UInt32 nb = abl->mNumberBuffers < channels ? abl->mNumberBuffers : channels;
        for (UInt32 c = 0; c < channels; c++) {
            const uint8_t *src = (c < nb) ? abl->mBuffers[c].mData : NULL;
            for (UInt32 i = 0; i < frames; i++) {
                dst[i * channels + c] = src ? sample_at(src, i, isFloat, isSigned, bits) : 0;
            }
        }
    }
    ring_push(dst, frames);
    return (int)frames;
}

static void *writer_main(void *arg) {
    (void)arg;
    size_t cap = 2048;
    int ch = g_ring.channels > 0 ? g_ring.channels : 1;
    int16_t *buf = calloc(cap * (size_t)ch, sizeof(int16_t));
    double last_log = now_sec();
    double win_sq = 0;
    size_t win_n = 0;
    int win_peak = 0;
    while (!g_stop || (atomic_load_explicit(&g_ring.write_frame, memory_order_acquire) !=
                       atomic_load_explicit(&g_ring.read_frame, memory_order_relaxed))) {
        size_t n = ring_pop(buf, cap);
        if (n == 0) {
            if (g_stop) break;
            usleep(5000);
            continue;
        }
        for (size_t i = 0; i < n * (size_t)ch; i++) {
            int v = buf[i];
            int a = v < 0 ? -v : v;
            if (a > win_peak) win_peak = a;
            if (a > g_peak_all) g_peak_all = a;
            double sq = (double)v * (double)v;
            win_sq += sq;
            g_sum_sq += sq;
            win_n++;
            g_sample_count++;
        }
        if (!g_probe) {
            if (!write_all(STDOUT_FILENO, buf, n * (size_t)ch * sizeof(int16_t))) {
                g_stop = 1;
                break;
            }
        }
        double t = now_sec();
        if (t - last_log >= 2.0) {
            double rms = win_n ? sqrt(win_sq / (double)win_n) : 0;
            unsigned long long pushed = atomic_load_explicit(&g_ring.pushed_frames, memory_order_relaxed);
            unsigned long long dropped = atomic_load_explicit(&g_ring.dropped_frames, memory_order_relaxed);
            caplog("电平 rms=%.0f peak=%d 原始峰值=%g 累计帧=%llu 丢帧=%llu", rms, win_peak, g_src_peak, pushed, dropped);
            last_log = t;
            win_sq = 0;
            win_n = 0;
            win_peak = 0;
        }
    }
    free(buf);
    return NULL;
}

static NSString *copy_string_prop(AudioObjectID obj, AudioObjectPropertySelector sel) {
    AudioObjectPropertyAddress addr = { sel, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    CFStringRef value = NULL;
    UInt32 size = sizeof(value);
    OSStatus st = AudioObjectGetPropertyData(obj, &addr, 0, NULL, &size, &value);
    if (st != noErr || !value) return nil;
    return (__bridge_transfer NSString *)value;
}

static UInt32 copy_u32_prop(AudioObjectID obj, AudioObjectPropertySelector sel, AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress addr = { sel, scope, kAudioObjectPropertyElementMain };
    UInt32 value = 0;
    UInt32 size = sizeof(value);
    AudioObjectGetPropertyData(obj, &addr, 0, NULL, &size, &value);
    return value;
}

static AudioObjectID default_device(AudioObjectPropertySelector sel) {
    return (AudioObjectID)copy_u32_prop(kAudioObjectSystemObject, sel, kAudioObjectPropertyScopeGlobal);
}

static BOOL is_aggregate(AudioObjectID dev) {
    return copy_u32_prop(dev, kAudioObjectPropertyClass, kAudioObjectPropertyScopeGlobal) == kAudioAggregateDeviceClassID;
}

static UInt32 channel_count(AudioObjectID dev, AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress addr = {
        kAudioDevicePropertyStreamConfiguration,
        scope,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    OSStatus st = AudioObjectGetPropertyDataSize(dev, &addr, 0, NULL, &size);
    if (st != noErr || size < sizeof(AudioBufferList)) return 0;
    AudioBufferList *abl = malloc(size);
    if (!abl) return 0;
    st = AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, abl);
    UInt32 ch = 0;
    if (st == noErr) {
        for (UInt32 i = 0; i < abl->mNumberBuffers; i++) ch += abl->mBuffers[i].mNumberChannels;
    }
    free(abl);
    return ch;
}

static double nominal_rate(AudioObjectID dev) {
    AudioObjectPropertyAddress addr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    Float64 rate = 0;
    UInt32 size = sizeof(rate);
    AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &rate);
    return rate;
}

static NSArray<NSString *> *aggregate_sub_uids(AudioObjectID dev) {
    if (!is_aggregate(dev)) return @[];
    AudioObjectPropertyAddress addr = {
        kAudioAggregateDevicePropertyFullSubDeviceList,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    CFArrayRef arr = NULL;
    UInt32 size = sizeof(arr);
    OSStatus st = AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &arr);
    if (st != noErr || !arr) return @[];
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    CFIndex n = CFArrayGetCount(arr);
    for (CFIndex i = 0; i < n; i++) {
        CFStringRef item = CFArrayGetValueAtIndex(arr, i);
        if (item) [out addObject:(__bridge NSString *)item];
    }
    CFRelease(arr);
    return out;
}

static NSArray<NSNumber *> *all_device_ids(void) {
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    OSStatus st = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &size);
    if (st != noErr || size < sizeof(AudioObjectID)) return @[];
    UInt32 count = size / sizeof(AudioObjectID);
    AudioObjectID *ids = calloc(count, sizeof(AudioObjectID));
    st = AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, ids);
    NSMutableArray<NSNumber *> *out = [NSMutableArray array];
    if (st == noErr) {
        for (UInt32 i = 0; i < count; i++) [out addObject:@(ids[i])];
    }
    free(ids);
    return out;
}

static void emit_json(NSDictionary *obj) {
    NSError *err = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:obj options:0 error:&err];
    if (!data) {
        caplog("JSON 编码失败: %s", err.localizedDescription.UTF8String);
        return;
    }
    write_all(STDOUT_FILENO, data.bytes, data.length);
    write_all(STDOUT_FILENO, "\n", 1);
}

static int cmd_list(void) {
    AudioObjectID defIn = default_device(kAudioHardwarePropertyDefaultInputDevice);
    AudioObjectID defOut = default_device(kAudioHardwarePropertyDefaultOutputDevice);
    AudioObjectID defSys = default_device(kAudioHardwarePropertyDefaultSystemOutputDevice);
    NSString *inName = @"";
    NSString *outName = @"";
    NSString *sysName = @"";
    NSString *inUID = @"";
    NSString *outUID = @"";
    NSString *sysUID = @"";
    for (NSNumber *num in all_device_ids()) {
        AudioObjectID dev = (AudioObjectID)num.unsignedIntValue;
        NSString *name = copy_string_prop(dev, kAudioObjectPropertyName) ?: @"";
        NSString *uid = copy_string_prop(dev, kAudioDevicePropertyDeviceUID) ?: @"";
        UInt32 transport = copy_u32_prop(dev, kAudioDevicePropertyTransportType, kAudioObjectPropertyScopeGlobal);
        BOOL agg = is_aggregate(dev);
        NSArray *subs = agg ? aggregate_sub_uids(dev) : @[];
        if (dev == defIn) { inName = name; inUID = uid; }
        if (dev == defOut) { outName = name; outUID = uid; }
        if (dev == defSys) { sysName = name; sysUID = uid; }
        emit_json(@{
            @"type": @"device",
            @"id": @(dev),
            @"name": name,
            @"uid": uid,
            @"input_channels": @(channel_count(dev, kAudioDevicePropertyScopeInput)),
            @"output_channels": @(channel_count(dev, kAudioDevicePropertyScopeOutput)),
            @"nominal_rate": @(nominal_rate(dev)),
            @"transport": fourcc_string(transport),
            @"is_aggregate": @(agg),
            @"subdevice_uids": subs,
            @"is_default_input": @(dev == defIn),
            @"is_default_output": @(dev == defOut),
            @"is_default_system_output": @(dev == defSys),
        });
    }
    emit_json(@{
        @"type": @"summary",
        @"default_input_name": inName,
        @"default_input_uid": inUID,
        @"default_output_name": outName,
        @"default_output_uid": outUID,
        @"default_system_output_name": sysName,
        @"default_system_output_uid": sysUID,
    });
    return 0;
}

static AudioObjectID find_device(NSString *name, NSString *uid, NSString **outName) {
    NSMutableArray<NSDictionary *> *matches = [NSMutableArray array];
    for (NSNumber *num in all_device_ids()) {
        AudioObjectID dev = (AudioObjectID)num.unsignedIntValue;
        NSString *devName = copy_string_prop(dev, kAudioObjectPropertyName) ?: @"";
        NSString *devUID = copy_string_prop(dev, kAudioDevicePropertyDeviceUID) ?: @"";
        BOOL uidMatch = uid.length && [devUID caseInsensitiveCompare:uid] == NSOrderedSame;
        BOOL nameExact = name.length && [devName caseInsensitiveCompare:name] == NSOrderedSame;
        BOOL nameHas = name.length && [devName rangeOfString:name options:NSCaseInsensitiveSearch].location != NSNotFound;
        if (uidMatch || nameExact || (!uid.length && nameHas)) {
            [matches addObject:@{@"id": num, @"name": devName, @"exact": @(uidMatch || nameExact)}];
        }
    }
    if (matches.count == 0) return 0;
    NSDictionary *chosen = nil;
    for (NSDictionary *m in matches) {
        if ([m[@"exact"] boolValue]) { chosen = m; break; }
    }
    if (!chosen) {
        if (matches.count > 1) {
            NSMutableArray *names = [NSMutableArray array];
            for (NSDictionary *m in matches) [names addObject:m[@"name"]];
            set_error([NSString stringWithFormat:@"设备名不唯一: %@", [names componentsJoinedByString:@", "]]);
            return 0;
        }
        chosen = matches[0];
    }
    if (outName) *outName = chosen[@"name"];
    return (AudioObjectID)[chosen[@"id"] unsignedIntValue];
}

static void push_planar_float(float * const *channels, UInt32 nch, UInt32 frames) {
    if (!channels || !g_scratch || frames == 0 || g_ring.channels <= 0) return;
    if ((int)nch != g_ring.channels) return;
    if (frames > g_scratch_frames) {
        if (!g_oversize_logged) {
            caplog("音频缓冲过大（%u 帧），已丢弃", frames);
            g_oversize_logged = 1;
        }
        return;
    }
    int ch = g_ring.channels;
    for (UInt32 i = 0; i < frames; i++) {
        for (UInt32 c = 0; c < nch; c++) {
            float x = channels[c] ? channels[c][i] : 0.f;
            if (x > 1.f) x = 1.f;
            if (x < -1.f) x = -1.f;
            float a = x < 0.f ? -x : x;
            if (a > g_src_peak) g_src_peak = a;
            g_scratch[i * ch + (int)c] = (int16_t)lrintf(x * 32767.f);
        }
    }
    ring_push(g_scratch, frames);
}

static BOOL start_engine_on_device(AudioObjectID dev) {
    g_engine = [AVAudioEngine new];
    AudioUnit unit = g_engine.inputNode.audioUnit;
    if (!unit) {
        set_error(@"无法取得输入音频单元");
        caplog("%s", g_error);
        return NO;
    }
    AudioOutputUnitStop(unit);
    AudioUnitUninitialize(unit);
    OSStatus st = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev, sizeof(dev));
    if (st != noErr) {
        set_error([NSString stringWithFormat:@"绑定采集设备失败: %@", os_status_string(st)]);
        caplog("%s", g_error);
        return NO;
    }
    AudioDeviceID bound = 0;
    UInt32 boundSize = sizeof(bound);
    AudioUnitGetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &bound, &boundSize);
    NSString *boundName = copy_string_prop(bound, kAudioObjectPropertyName);
    caplog("引擎绑定设备 id=%u name=%s 目标=%u", bound, boundName.UTF8String ?: "", dev);
    if (bound != dev) {
        set_error([NSString stringWithFormat:@"引擎绑到了「%@」，不是采集 tap", boundName ?: @"未知设备"]);
        caplog("%s", g_error);
        return NO;
    }
    AudioUnitInitialize(unit);
    AVAudioFormat *fmt = [g_engine.inputNode outputFormatForBus:0];
    if (fmt && g_wanted_channels > fmt.channelCount && g_wanted_channels <= 8) {
        AVAudioFormat *wider = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                                sampleRate:fmt.sampleRate
                                                                  channels:g_wanted_channels
                                                               interleaved:NO];
        if (wider) {
            caplog("引擎是 %u 声道，改用 tap 的 %u 声道", (unsigned)fmt.channelCount, g_wanted_channels);
            fmt = wider;
        }
    }
    if (!fmt || fmt.sampleRate <= 0 || fmt.channelCount == 0 || fmt.channelCount > 8) {
        set_error(@"采集格式无效");
        caplog("%s", g_error);
        return NO;
    }
    memset(&g_fmt, 0, sizeof(g_fmt));
    g_fmt.mSampleRate = fmt.sampleRate;
    g_fmt.mChannelsPerFrame = (UInt32)fmt.channelCount;
    g_fmt.mBitsPerChannel = 32;
    g_fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    g_fmt_ready = 1;
    size_t cap = (size_t)fmt.sampleRate * 8;
    if (cap < 48000) cap = 48000;
    ring_init(cap, (int)fmt.channelCount);
    caplog("采集格式 %.0f Hz, %u 声道, engine float", fmt.sampleRate, (unsigned)fmt.channelCount);
    @try {
        [g_engine.inputNode installTapOnBus:0 bufferSize:1024 format:fmt block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
            (void)when;
            if (g_stop || !buffer.floatChannelData) return;
            push_planar_float(buffer.floatChannelData, (UInt32)buffer.format.channelCount, (UInt32)buffer.frameLength);
        }];
    } @catch (NSException *ex) {
        set_error([NSString stringWithFormat:@"安装采集回调失败: %@", ex.reason ?: ex.name]);
        caplog("%s", g_error);
        return NO;
    }
    NSError *err = nil;
    if (![g_engine startAndReturnError:&err]) {
        set_error(err.localizedDescription ?: @"启动音频引擎失败");
        caplog("%s", g_error);
        return NO;
    }
    return YES;
}

static void stop_engine(void) {
    AVAudioEngine *engine = g_engine;
    g_engine = nil;
    if (!engine) return;
    [engine.inputNode removeTapOnBus:0];
    [engine stop];
}

static BOOL ensure_input_permission(void) {
    AVAuthorizationStatus auth = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    caplog("麦克风权限状态=%ld（0 未决定，2 拒绝，3 已允许）", (long)auth);
    if (auth == AVAuthorizationStatusAuthorized) return YES;
    if (auth == AVAuthorizationStatusDenied || auth == AVAuthorizationStatusRestricted) {
        caplog("麦克风权限被拒绝。读取 BlackHole 这类输入设备需要它，否则系统只会给出静音。请到系统设置 > 隐私与安全性 > 麦克风，允许当前应用。");
        return NO;
    }
    caplog("正在请求麦克风权限。请在弹窗里允许。读取输入设备需要这项权限；系统 tap 不需要。");
    __block atomic_int done = 0;
    __block BOOL granted = NO;
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL ok) {
        granted = ok;
        atomic_store(&done, 1);
    }];
    double deadline = now_sec() + 45;
    while (atomic_load(&done) == 0 && !g_stop && now_sec() < deadline) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.2, false);
    }
    caplog("麦克风权限请求结果=%s", granted ? "允许" : "未允许");
    return granted;
}

static NSString *default_output_uid(NSString **outName) {
    AudioObjectID dev = default_device(kAudioHardwarePropertyDefaultSystemOutputDevice);
    if (!dev) dev = default_device(kAudioHardwarePropertyDefaultOutputDevice);
    if (!dev) return nil;
    if (outName) *outName = copy_string_prop(dev, kAudioObjectPropertyName);
    return copy_string_prop(dev, kAudioDevicePropertyDeviceUID);
}

static void log_tap_format(AudioObjectID tapID) {
    AudioStreamBasicDescription fmt = {0};
    AudioObjectPropertyAddress addr = { kAudioTapPropertyFormat, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    UInt32 size = sizeof(fmt);
    OSStatus st = AudioObjectGetPropertyData(tapID, &addr, 0, NULL, &size, &fmt);
    if (st == noErr && fmt.mChannelsPerFrame > 0) {
        g_wanted_channels = fmt.mChannelsPerFrame;
        caplog("tap 格式 %.0f Hz %u 声道", fmt.mSampleRate, fmt.mChannelsPerFrame);
    } else {
        caplog("读取 tap 格式失败: %s", os_status_string(st).UTF8String);
    }
}

static void log_silence_hint(void) {
    char path[PROC_PIDPATHINFO_MAXSIZE];
    pid_t pid = getppid();
    if (proc_pidpath(pid, path, sizeof(path)) <= 0) {
        snprintf(path, sizeof(path), "当前终端");
    }
    bool allowed = CGPreflightScreenCaptureAccess();
    caplog("开头没有波形。如果耳机里正在放歌仍一直是 0，请到系统设置 > 隐私与安全性 > 屏幕录制，允许 %s 后完全退出再运行。当前屏幕录制权限=%s",
           path, allowed ? "已允许" : "未允许");
    if (!allowed) CGRequestScreenCaptureAccess();
}

static BOOL setup_system_tap_kind(BOOL globalTap, NSString *forcedUID, NSString *forcedName) {
    NSString *outputName = forcedName;
    NSString *outputUID = forcedUID;
    if (!globalTap && !outputUID.length) {
        outputUID = default_output_uid(&outputName);
    }
    caplog("%s「%s」（不静音，不修改默认设备）",
           globalTap ? "正在采集全部程序输出，当前系统输出是" : "正在采集输出",
           outputName.UTF8String ?: outputUID.UTF8String ?: "未知设备");
    CATapDescription *tap = nil;
    if (!globalTap && outputUID.length) {
        tap = [[CATapDescription alloc] initExcludingProcesses:@[] andDeviceUID:outputUID withStream:0];
    }
    if (!tap) {
        if (!globalTap) caplog("无法按输出设备建 tap，改为采集全部进程输出");
        tap = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:@[]];
    }
    tap.name = @"sysaudio-stt";
    tap.privateTap = YES;
    tap.muteBehavior = CATapUnmuted;
    if (!tap.UUID) tap.UUID = [NSUUID UUID];
    NSString *tapUID = tap.UUID.UUIDString;
    caplog("静音策略=%ld（0 表示播放保持打开）", (long)tap.muteBehavior);
    if (tap.muteBehavior != CATapUnmuted) {
        set_error(@"拒绝启动：tap 静音策略不是 unmuted");
        caplog("%s", g_error);
        return NO;
    }
    AudioObjectID tapID = 0;
    OSStatus st = AudioHardwareCreateProcessTap(tap, &tapID);
    if (st != noErr || tapID == 0) {
        set_error([NSString stringWithFormat:@"创建系统音频 tap 失败: %@。若系统弹出权限请允许；也可改用 BlackHole：--source device --device \"BlackHole 2ch\"", os_status_string(st)]);
        caplog("%s", g_error);
        return NO;
    }
    g_tap = tapID;
    caplog("tap 已创建 id=%u uid=%s", tapID, tapUID.UTF8String);
    log_tap_format(tapID);
    NSString *aggUID = [[NSUUID UUID] UUIDString];
    NSDictionary *desc = @{
        [NSString stringWithUTF8String:kAudioAggregateDeviceNameKey]: @"sysaudio-stt",
        [NSString stringWithUTF8String:kAudioAggregateDeviceUIDKey]: aggUID,
        [NSString stringWithUTF8String:kAudioAggregateDeviceIsPrivateKey]: @1,
        [NSString stringWithUTF8String:kAudioAggregateDeviceIsStackedKey]: @0,
        [NSString stringWithUTF8String:kAudioAggregateDeviceTapAutoStartKey]: @0,
        [NSString stringWithUTF8String:kAudioAggregateDeviceTapListKey]: @[
            @{
                [NSString stringWithUTF8String:kAudioSubTapUIDKey]: tapUID,
                [NSString stringWithUTF8String:kAudioSubTapDriftCompensationKey]: @1,
            }
        ],
    };
    AudioObjectID agg = 0;
    st = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)desc, &agg);
    if (st != noErr || agg == 0) {
        set_error([NSString stringWithFormat:@"创建私有聚合设备失败: %@", os_status_string(st)]);
        caplog("%s", g_error);
        return NO;
    }
    g_aggregate = agg;
    g_capture_device = agg;
    caplog("私有聚合设备已创建 id=%u", agg);
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, false);
    if (!start_engine_on_device(agg)) return NO;
    set_label(g_device_label, sizeof(g_device_label), outputName ?: @"system-output");
    snprintf(g_source_label, sizeof(g_source_label), "system");
    return YES;
}

static void teardown_tap_only(void) {
    stop_engine();
    if (g_aggregate) {
        AudioHardwareDestroyAggregateDevice(g_aggregate);
        g_aggregate = 0;
    }
    if (g_tap) {
        AudioHardwareDestroyProcessTap(g_tap);
        g_tap = 0;
    }
    g_capture_device = 0;
    g_src_peak = 0;
    g_wanted_channels = 0;
}

static BOOL wait_for_signal(double seconds) {
    double start = now_sec();
    while (!g_stop && now_sec() - start < seconds) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, false);
        if (g_src_peak > 0.0005f) return YES;
    }
    return g_src_peak > 0.0005f;
}

static BOOL setup_system_tap(void) {
    if (!setup_system_tap_kind(NO, nil, nil)) return NO;
    if (wait_for_signal(1.2)) return YES;
    caplog("开头 1.2 秒没有波形，正在重试全部进程采集");
    teardown_tap_only();
    if (!setup_system_tap_kind(YES, nil, nil)) return NO;
    if (!wait_for_signal(1.2)) log_silence_hint();
    return YES;
}

static void reset_levels(void) {
    g_sum_sq = 0;
    g_sample_count = 0;
    g_peak_all = 0;
    g_src_peak = 0;
    g_diag_logged = 0;
}

static void release_graph(void) {
    g_stop = 1;
    stop_engine();
    if (g_writer_started) {
        pthread_join(g_writer, NULL);
        g_writer_started = 0;
    }
    if (g_aggregate) {
        AudioHardwareDestroyAggregateDevice(g_aggregate);
        g_aggregate = 0;
    }
    if (g_tap) {
        AudioHardwareDestroyProcessTap(g_tap);
        g_tap = 0;
    }
    g_capture_device = 0;
    ring_free();
    g_fmt_ready = 0;
    g_wanted_channels = 0;
    g_stop = 0;
    reset_levels();
}

static BOOL setup_selected_output(NSString *name, NSString *uid) {
    NSString *found = nil;
    AudioObjectID dev = find_device(name, uid, &found);
    if (!dev) {
        if (!g_error[0]) set_error([NSString stringWithFormat:@"找不到输出设备: %@", name ?: uid ?: @"(未指定)"]);
        caplog("%s", g_error);
        return NO;
    }
    if (channel_count(dev, kAudioDevicePropertyScopeOutput) == 0) {
        set_error([NSString stringWithFormat:@"设备「%@」没有输出声道", found]);
        caplog("%s", g_error);
        return NO;
    }
    NSString *useUID = uid.length ? uid : copy_string_prop(dev, kAudioDevicePropertyDeviceUID);
    caplog("按选择监听「%s」，不改默认设备", found.UTF8String ?: "");
    return setup_system_tap_kind(NO, useUID, found);
}

static BOOL setup_device(NSString *name, NSString *uid) {
    NSString *found = nil;
    AudioObjectID dev = find_device(name, uid, &found);
    if (!dev) {
        if (!g_error[0]) set_error([NSString stringWithFormat:@"找不到输入设备: %@", name ?: uid ?: @"(未指定)"]);
        caplog("%s", g_error);
        return NO;
    }
    UInt32 inCh = channel_count(dev, kAudioDevicePropertyScopeInput);
    if (inCh == 0) {
        set_error([NSString stringWithFormat:@"设备「%@」没有输入声道", found]);
        caplog("%s", g_error);
        return NO;
    }
    caplog("打开输入设备「%s」（不占用设备，不修改系统默认设备）", found.UTF8String);
    if (!ensure_input_permission()) {
        set_error(@"没有麦克风权限，不能读取输入设备");
        caplog("%s", g_error);
        return NO;
    }
    g_capture_device = dev;
    if (!start_engine_on_device(dev)) return NO;
    set_label(g_device_label, sizeof(g_device_label), found);
    snprintf(g_source_label, sizeof(g_source_label), "device");
    return YES;
}

@interface SCKOutput : NSObject <SCStreamOutput, SCStreamDelegate>
@end

@implementation SCKOutput
- (void)stream:(SCStream *)stream didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer ofType:(SCStreamOutputType)type {
    (void)stream;
    if (type != SCStreamOutputTypeAudio || g_stop) return;
    if (!CMSampleBufferDataIsReady(sampleBuffer)) return;
    CMFormatDescriptionRef fd = CMSampleBufferGetFormatDescription(sampleBuffer);
    const AudioStreamBasicDescription *asbd = fd ? CMAudioFormatDescriptionGetStreamBasicDescription(fd) : NULL;
    if (asbd && !g_fmt_ready) {
        g_fmt = *asbd;
        g_fmt_ready = 1;
    }
    const AudioStreamBasicDescription *use = g_fmt_ready ? &g_fmt : asbd;
    if (!use) return;
    size_t needed = 0;
    OSStatus st = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer, &needed, NULL, 0, NULL, NULL, 0, NULL);
    if (needed == 0) return;
    AudioBufferList *abl = malloc(needed);
    if (!abl) return;
    CMBlockBufferRef block = NULL;
    st = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer, NULL, abl, needed, NULL, NULL,
        kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, &block);
    if (st == noErr) push_from_abl(abl, use);
    if (block) CFRelease(block);
    free(abl);
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    (void)stream;
    if (error) {
        set_error(error.localizedDescription);
        caplog("屏幕采集中断: %s", g_error);
    }
    g_stop = 1;
}
@end

static BOOL setup_sckit(void) {
    caplog("正在通过屏幕采集接口抓系统音频（不采集麦克风）。若弹出屏幕录制权限，请允许。菜单栏可能出现录制指示。");
    atomic_store(&g_sckit_phase, 0);
    g_sck = [SCKOutput new];
    [SCShareableContent getShareableContentExcludingDesktopWindows:YES
                                             onScreenWindowsOnly:YES
                                               completionHandler:^(SCShareableContent *content, NSError *error) {
        if (error || content.displays.count == 0) {
            set_error(error.localizedDescription ?: @"没有可用显示器，无法建立系统音频采集");
            caplog("%s", g_error);
            atomic_store(&g_sckit_phase, -1);
            return;
        }
        CGDirectDisplayID mainID = CGMainDisplayID();
        SCDisplay *display = content.displays.firstObject;
        for (SCDisplay *d in content.displays) {
            if (d.displayID == mainID) { display = d; break; }
        }
        SCContentFilter *filter = [[SCContentFilter alloc] initWithDisplay:display excludingWindows:@[]];
        SCStreamConfiguration *config = [SCStreamConfiguration new];
        config.capturesAudio = YES;
        config.excludesCurrentProcessAudio = YES;
        config.sampleRate = 48000;
        config.channelCount = 2;
        config.width = 64;
        config.height = 64;
        config.minimumFrameInterval = CMTimeMake(1, 1);
        config.showsCursor = NO;
        config.queueDepth = 3;
        if (@available(macOS 15.0, *)) {
            config.captureMicrophone = NO;
        }
        memset(&g_fmt, 0, sizeof(g_fmt));
        g_fmt.mSampleRate = 48000;
        g_fmt.mFormatID = kAudioFormatLinearPCM;
        g_fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        g_fmt.mBitsPerChannel = 32;
        g_fmt.mChannelsPerFrame = 2;
        g_fmt.mBytesPerFrame = 8;
        g_fmt.mFramesPerPacket = 1;
        g_fmt.mBytesPerPacket = 8;
        g_fmt_ready = 1;
        ring_init((size_t)g_fmt.mSampleRate * 8, 2);
        NSError *addErr = nil;
        g_stream = [[SCStream alloc] initWithFilter:filter configuration:config delegate:g_sck];
        BOOL added = [g_stream addStreamOutput:g_sck type:SCStreamOutputTypeAudio sampleHandlerQueue:dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0) error:&addErr];
        if (!added) {
            set_error(addErr.localizedDescription ?: @"添加音频输出失败");
            caplog("%s", g_error);
            atomic_store(&g_sckit_phase, -1);
            return;
        }
        atomic_store(&g_sckit_phase, 1);
        [g_stream startCaptureWithCompletionHandler:^(NSError *startErr) {
            if (startErr) {
                set_error(startErr.localizedDescription ?: @"启动屏幕音频采集失败");
                caplog("%s", g_error);
                atomic_store(&g_sckit_phase, -1);
                return;
            }
            caplog("屏幕音频采集已启动（麦克风关闭）");
            atomic_store(&g_sckit_phase, 2);
        }];
    }];
    double deadline = now_sec() + 20;
    while (!g_stop) {
        int phase = atomic_load(&g_sckit_phase);
        if (phase == 2) break;
        if (phase < 0) return NO;
        if (now_sec() > deadline) {
            set_error(@"等待屏幕采集权限或启动超时");
            caplog("%s", g_error);
            return NO;
        }
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, false);
    }
    if (g_stop) return NO;
    set_label(g_device_label, sizeof(g_device_label), @"sckit");
    snprintf(g_source_label, sizeof(g_source_label), "sckit");
    caplog("采集格式 %.0f Hz, %u 声道, %u bit, flags=0x%x",
         g_fmt.mSampleRate, g_fmt.mChannelsPerFrame, g_fmt.mBitsPerChannel, g_fmt.mFormatFlags);
    return YES;
}

static void stop_sckit(void) {
    SCStream *stream = g_stream;
    g_stream = nil;
    if (!stream) return;
    __block atomic_int done = 0;
    [stream stopCaptureWithCompletionHandler:^(NSError *error) {
        (void)error;
        atomic_store(&done, 1);
    }];
    double deadline = now_sec() + 2;
    while (atomic_load(&done) == 0 && now_sec() < deadline) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, false);
    }
    g_sck = nil;
}

static void cleanup(void) {
    if (g_cleanup_done) return;
    g_cleanup_done = 1;
    g_stop = 1;
    stop_engine();
    stop_sckit();
    if (g_capture_device && g_proc) {
        AudioDeviceStop(g_capture_device, g_proc);
        AudioDeviceDestroyIOProcID(g_capture_device, g_proc);
        g_proc = NULL;
    }
    if (g_writer_started) {
        pthread_join(g_writer, NULL);
        g_writer_started = 0;
    }
    if (g_aggregate) {
        OSStatus st = AudioHardwareDestroyAggregateDevice(g_aggregate);
        if (st != noErr) caplog("销毁聚合设备失败: %s", os_status_string(st).UTF8String);
        g_aggregate = 0;
    }
    if (g_tap) {
        OSStatus st = AudioHardwareDestroyProcessTap(g_tap);
        if (st != noErr) caplog("销毁 tap 失败: %s", os_status_string(st).UTF8String);
        g_tap = 0;
    }
    g_capture_device = 0;
    ring_free();
    caplog("采集资源已释放");
}

static BOOL write_header(void) {
    char buf[640];
    int n = snprintf(buf, sizeof(buf),
                     "SYSAUDIO1\nrate=%.0f\nchannels=%d\nencoding=s16le\nsource=%s\ndevice=%s\n\n",
                     g_fmt.mSampleRate, g_ring.channels, g_source_label, g_device_label);
    if (n < 0 || n >= (int)sizeof(buf)) return NO;
    return write_all(STDOUT_FILENO, buf, (size_t)n);
}

static void write_probe_json(BOOL ok, double elapsed) {
    int ch = g_ring.channels > 0 ? g_ring.channels : 1;
    unsigned long long samples = g_sample_count;
    double rms = samples ? sqrt(g_sum_sq / (double)samples) : 0;
    NSDictionary *obj = @{
        @"ok": @(ok),
        @"error": ok ? @"" : [NSString stringWithUTF8String:g_error[0] ? g_error : "采集失败"],
        @"source": [NSString stringWithUTF8String:g_source_label],
        @"device": [NSString stringWithUTF8String:g_device_label],
        @"rate": @(g_fmt.mSampleRate),
        @"channels": @(g_ring.channels),
        @"encoding": @"s16le",
        @"seconds": @(elapsed),
        @"samples": @(samples),
        @"frames": @(ch ? samples / (unsigned long long)ch : 0),
        @"dropped_frames": @(atomic_load(&g_ring.dropped_frames)),
        @"buffers": @(atomic_load(&g_ring.buffers)),
        @"rms_peak": @(g_peak_all),
        @"rms_mean": @(rms),
    };
    emit_json(obj);
}

static NSString *host_app_name(void) {
    pid_t pid = getpid();
    NSString *found = nil;
    for (int i = 0; i < 12 && pid > 1; i++) {
        char path[PROC_PIDPATHINFO_MAXSIZE];
        if (proc_pidpath(pid, path, sizeof(path)) > 0) {
            NSString *full = [NSString stringWithUTF8String:path];
            NSRange marker = [full rangeOfString:@".app/Contents/MacOS/"];
            if (marker.location != NSNotFound) {
                found = [[full substringToIndex:marker.location] lastPathComponent];
            }
        }
        struct proc_bsdinfo info;
        int got = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, sizeof(info));
        if (got <= 0 || info.pbi_ppid == 0 || info.pbi_ppid == pid) break;
        pid = info.pbi_ppid;
    }
    return found ?: @"当前终端";
}

static NSArray<NSString *> *playing_process_names(void) {
    AudioObjectPropertyAddress addr = { kAudioHardwarePropertyProcessObjectList, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &size) != noErr || size < sizeof(AudioObjectID)) return @[];
    UInt32 count = size / sizeof(AudioObjectID);
    AudioObjectID *ids = calloc(count, sizeof(AudioObjectID));
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, ids) != noErr) {
        free(ids);
        return @[];
    }
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (UInt32 i = 0; i < count; i++) {
        AudioObjectPropertyAddress pa = { kAudioProcessPropertyIsRunningOutput, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
        UInt32 running = 0;
        UInt32 sz = sizeof(running);
        if (AudioObjectGetPropertyData(ids[i], &pa, 0, NULL, &sz, &running) != noErr || !running) continue;
        pa.mSelector = kAudioProcessPropertyBundleID;
        CFStringRef bid = NULL;
        sz = sizeof(bid);
        NSString *label = nil;
        if (AudioObjectGetPropertyData(ids[i], &pa, 0, NULL, &sz, &bid) == noErr && bid) {
            label = [(__bridge NSString *)bid lastPathComponent];
            CFRelease(bid);
        }
        if (label.length) [names addObject:label];
    }
    free(ids);
    return names;
}

static int cmd_scan(double seconds) {
    if (seconds < 0.3) seconds = 0.3;
    if (seconds > 3.0) seconds = 3.0;
    g_probe = 1;
    NSMutableArray<NSDictionary *> *outputs = [NSMutableArray array];
    for (NSNumber *num in all_device_ids()) {
        AudioObjectID dev = (AudioObjectID)num.unsignedIntValue;
        if (channel_count(dev, kAudioDevicePropertyScopeOutput) == 0) continue;
        NSString *name = copy_string_prop(dev, kAudioObjectPropertyName) ?: @"";
        NSString *uid = copy_string_prop(dev, kAudioDevicePropertyDeviceUID) ?: @"";
        BOOL defOut = dev == default_device(kAudioHardwarePropertyDefaultOutputDevice);
        BOOL defSys = dev == default_device(kAudioHardwarePropertyDefaultSystemOutputDevice);
        [outputs addObject:@{
            @"id": @(dev),
            @"name": name,
            @"uid": uid,
            @"is_default_output": @(defOut),
            @"is_default_system_output": @(defSys),
            @"output_channels": @(channel_count(dev, kAudioDevicePropertyScopeOutput)),
        }];
    }
    caplog("开始测量 %lu 路输出，每路约 %.1f 秒。不改默认设备，不静音。", (unsigned long)outputs.count, seconds);
    int maxRMS = 0;
    for (NSDictionary *item in outputs) {
        NSString *name = item[@"name"];
        NSString *uid = item[@"uid"];
        caplog("正在测量「%s」", name.UTF8String ?: "");
        NSString *err = @"";
        double rms = 0;
        int peak = 0;
        if (!setup_system_tap_kind(NO, uid, name)) {
            err = [NSString stringWithUTF8String:g_error[0] ? g_error : "测量失败"];
        } else if (pthread_create(&g_writer, NULL, writer_main, NULL) != 0) {
            err = @"创建测量线程失败";
        } else {
            g_writer_started = 1;
            double start = now_sec();
            while (!g_stop && now_sec() - start < seconds) {
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, false);
            }
            g_stop = 1;
            pthread_join(g_writer, NULL);
            g_writer_started = 0;
            g_stop = 0;
            if (g_sample_count > 0) rms = sqrt(g_sum_sq / (double)g_sample_count);
            peak = g_peak_all;
            if ((int)llround(rms) > maxRMS) maxRMS = (int)llround(rms);
        }
        release_graph();
        g_error[0] = 0;
        emit_json(@{
            @"type": @"output",
            @"name": name ?: @"",
            @"uid": uid ?: @"",
            @"rms": @((int)llround(rms)),
            @"peak": @(peak),
            @"is_default_output": item[@"is_default_output"],
            @"is_default_system_output": item[@"is_default_system_output"],
            @"output_channels": item[@"output_channels"],
            @"error": err,
        });
    }
    NSArray<NSString *> *players = playing_process_names();
    if (maxRMS == 0 && players.count > 0) {
        NSString *app = host_app_name();
        BOOL allowed = CGPreflightScreenCaptureAccess();
        NSString *hint = [NSString stringWithFormat:@"%@ 正在播放，但 %@ 启动的采集读到的全是 0。请往下滚到「仅系统录音」，确认 %@ 在列表里并且开关打开。上面的录屏开关不够。然后完全退出 %@ 再打开，重新运行。",
                          [players componentsJoinedByString:@"、"], app, app, app];
        caplog("%s", hint.UTF8String);
        if (!allowed) CGRequestScreenCaptureAccess();
        emit_json(@{ @"type": @"hint", @"app": app, @"message": hint, @"players": players });
    }
    return 0;
}

static void print_help(void) {
    fputs(
        "用法:\n"
        "  sysaudio-capture list\n"
        "  sysaudio-capture scan [--seconds N]\n"
        "  sysaudio-capture capture --source system|device|sckit [--device 名称] [--uid UID] [--seconds N]\n"
        "  sysaudio-capture probe   --source system|device|sckit [--device 名称] [--uid UID] [--seconds N]\n"
        "\n"
        "system  捕获指定输出设备上的程序声音。给出 --uid 或 --device 时只听那一路。不改默认设备，不静音。\n"
        "device  读取指定输入设备（例如 BlackHole 2ch）。不会自动改用麦克风。\n"
        "sckit   用屏幕采集接口抓系统音频，不采集麦克风，可能出现录制指示。\n",
        stdout);
}

static int run_capture(const char *source, NSString *device, NSString *uid) {
    BOOL ok = NO;
    if (strcmp(source, "system") == 0) {
        if ((device && device.length) || (uid && uid.length)) ok = setup_selected_output(device, uid);
        else ok = setup_system_tap();
    }
    else if (strcmp(source, "device") == 0) ok = setup_device(device, uid);
    else if (strcmp(source, "sckit") == 0) ok = setup_sckit();
    else {
        set_error([NSString stringWithFormat:@"未知 source: %s", source]);
        caplog("%s", g_error);
        return 2;
    }
    if (!ok || g_stop) {
        if (g_probe) write_probe_json(NO, 0);
        cleanup();
        return 1;
    }
    if (!g_probe) {
        if (!write_header()) {
            set_error(@"写入音频头失败");
            caplog("%s", g_error);
            cleanup();
            return 1;
        }
    }
    if (pthread_create(&g_writer, NULL, writer_main, NULL) != 0) {
        set_error(@"创建写入线程失败");
        caplog("%s", g_error);
        cleanup();
        return 1;
    }
    g_writer_started = 1;
    caplog("已开始采集 source=%s device=%s", g_source_label, g_device_label);
    double start = now_sec();
    while (!g_stop) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.2, false);
        if (g_seconds > 0 && now_sec() - start >= g_seconds) break;
    }
    double elapsed = now_sec() - start;
    g_stop = 1;
    cleanup();
    if (g_probe) write_probe_json(YES, elapsed);
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        signal(SIGPIPE, SIG_IGN);
        if (argc < 2 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0) {
            print_help();
            return argc < 2 ? 2 : 0;
        }
        const char *cmd = argv[1];
        const char *source = "system";
        const char *device = NULL;
        const char *uid = NULL;
        double seconds = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) source = argv[++i];
            else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
            else if (strcmp(argv[i], "--uid") == 0 && i + 1 < argc) uid = argv[++i];
            else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) seconds = atof(argv[++i]);
            else {
                caplog("未知参数: %s", argv[i]);
                print_help();
                return 2;
            }
        }
        g_seconds = seconds;
        if (strcmp(cmd, "list") == 0) return cmd_list();
        if (strcmp(cmd, "scan") == 0) return cmd_scan(seconds > 0 ? seconds : 0.8);
        if (strcmp(cmd, "capture") != 0 && strcmp(cmd, "probe") != 0) {
            caplog("未知命令: %s", cmd);
            print_help();
            return 2;
        }
        g_probe = strcmp(cmd, "probe") == 0;
        if (strcmp(source, "device") == 0 && !device && !uid) {
            caplog("device 模式必须指定 --device 或 --uid，不会自动改用麦克风");
            return 2;
        }
        NSString *deviceName = device ? [NSString stringWithUTF8String:device] : nil;
        NSString *deviceUID = uid ? [NSString stringWithUTF8String:uid] : nil;
        return run_capture(source, deviceName, deviceUID);
    }
}
