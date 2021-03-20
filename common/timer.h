#ifndef __TIME_UTILS__
#define __TIME_UTILS__
#include <sys/time.h>


typedef struct {
    struct timeval begin;
    struct timeval end;
} ptimer_t;



inline void timer_start(ptimer_t *tp){
    gettimeofday(&(tp->begin), 0);
}


inline void timer_stop(ptimer_t *tp){
    gettimeofday(&(tp->end), 0);
}


double timer_elapsed(ptimer_t tp){
    long seconds = tp.end.tv_sec - tp.begin.tv_sec;
    long microseconds = tp.end.tv_usec - tp.begin.tv_usec;
    double elapsed = seconds * 1e3 + microseconds * 1e-3;
    return elapsed; 
}
#endif