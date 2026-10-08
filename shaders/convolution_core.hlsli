// Shared compact NCHW8 convolution arithmetic. The including shader supplies
// ACTIVATION_WORD(address) and WEIGHT_WORD(address).
#ifndef KERNEL_SIZE
#define KERNEL_SIZE 3
#endif
uint conv_load_half(uint word,uint address) { return (word>>((address&2u)*8u))&65535u; }
uint activation(uint c,int y,int x,uint width,uint height,uint address,uint e5) {
    if(x<0 || y<0 || x>=int(width) || y>=int(height)) return 0;
    uint index=(((c/8)*height+uint(y))*width+uint(x))*8+c%8;
    if(e5!=0) {
        uint a=address+index;
        return ((ACTIVATION_WORD(a&~3u)>>((a&3u)*8u))&255u)<<7;
    }
    uint a=address+2*index;
    return conv_load_half(ACTIVATION_WORD(a&~3u),a);
}
uint weight_half(uint address) { return conv_load_half(WEIGHT_WORD(address&~3u),address); }
uint convolution(uint output,int y,int x,uint width,uint height,uint channels,uint outputs,
                 uint e5,uint input_offset,uint weight_offset,uint bias_offset) {
    uint accumulated=0;
    uint lane=(output%8)/2;
    [loop] for(uint base=0;base<channels;base+=16) {
        [loop] for(uint kx=0;kx<KERNEL_SIZE;++kx) {
            [loop] for(uint ky=0;ky<KERNEL_SIZE;++ky) {
                precise float d=fp_h2f(accumulated);
                [loop] for(uint pair=0;pair<8;++pair) {
                    uint c=base+((lane+pair)%4)*2+(pair>=4?8:0);
                    uint wi=((((c/8)*KERNEL_SIZE+ky)*KERNEL_SIZE+kx)*outputs+output)*8+c%8;
                    uint a=activation(c,y+int(ky)-KERNEL_SIZE/2,x+int(kx)-KERNEL_SIZE/2,width,height,input_offset,e5);
                    uint b=weight_half(weight_offset+2*wi);
                    d=fp_fmix(a,b,d);
                    a=activation(c+1,y+int(ky)-KERNEL_SIZE/2,x+int(kx)-KERNEL_SIZE/2,width,height,input_offset,e5);
                    b=weight_half(weight_offset+2*(wi+1));
                    if(pair==7) accumulated=fp_mixlo(a,b,d);
                    else d=fp_fmix(a,b,d);
                }
            }
        }
    }
    uint r=fp_add16(accumulated,weight_half(bias_offset+2*output));
    if((r&0x7fffu)>0x7c00u) return 0x7fffu;
    return (r&0x8000u)?0:r;
}
// max-number on nonnegative post-ReLU half encodings; retain NaN if both are NaN.
uint max_half(uint a,uint b) {
    if(a>0x7c00u) return b;
    if(b>0x7c00u) return a;
    return max(a,b);
}
