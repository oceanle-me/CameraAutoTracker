#include "tracking.h"

/*return a center (x axis) of the best object that is familiar with the old object  */
int tracking(float prev_location, float curr_location_detect) {
    static int myqueue[LEN_QUE]={};

    float average=0;
    for(int i=0,sum =0; i<LEN_QUE;i++){
        sum += myqueue[i];
        average = sum/(i+1);
    }

    for (int i=LEN_QUE-1;i>0;i++){
        myqueue[i]=myqueue[i-1];
    }

    return 0;

}
