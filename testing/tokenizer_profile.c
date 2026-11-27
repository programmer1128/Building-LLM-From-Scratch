#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

// Linux/glibc-only probe. Counts calls through these entry points on the
// calling thread. Python's small-object allocator is outside this scope.
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
static _Thread_local int active;
static _Thread_local uint64_t calls;
static _Thread_local int fds[3];
static _Thread_local int errors[3];

void *malloc(size_t n)
{
    calls += active;
    return __libc_malloc(n);
}
void *calloc(size_t n, size_t s)
{
    calls += active;
    return __libc_calloc(n, s);
}
void *realloc(void *p, size_t n)
{
    calls += active;
    return __libc_realloc(p, n);
}
void *aligned_alloc(size_t a, size_t n)
{
    calls += active;
    return __libc_memalign(a, n);
}
int posix_memalign(void **p, size_t a, size_t n)
{
    if (a < sizeof(void *) || (a & (a - 1)))
    {
        return EINVAL;
    }
    calls += active;
    void *result = __libc_memalign(a, n);
    if (!result)
    {
        return ENOMEM;
    }
    *p = result;
    return 0;
}

void token_profile_start(void)
{
    const uint64_t events[] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_CACHE_MISSES,
                               PERF_COUNT_HW_BRANCH_MISSES};
    for (int i = 0; i < 3; ++i)
    {
        struct perf_event_attr attr = {0};
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof attr;
        attr.config = events[i];
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        fds[i] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
        errors[i] = fds[i] < 0 ? errno : 0;
        if (fds[i] >= 0)
        {
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    calls = 0;
    active = 1;
}

void token_profile_stop(uint64_t *out)
{
    active = 0;
    out[0] = calls;
    for (int i = 0; i < 3; ++i)
    {
        uint64_t data[3] = {0};
        if (fds[i] >= 0)
        {
            ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
            if (read(fds[i], data, sizeof data) != sizeof data || !data[2])
            {
                errors[i] = EIO;
            }
            close(fds[i]);
        }
        out[1 + i] = errors[i];
        out[4 + i] = errors[i] ? 0 : (uint64_t)((long double)data[0] * data[1] / data[2]);
    }
}
