[[vk::binding(0,0)]] ByteAddressBuffer Source;
[[vk::binding(1,0)]] RWByteAddressBuffer Destination;
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint value=Source.Load(id.x*4);
    uint lane=WaveGetLaneIndex();
    Destination.Store4(id.x*32,uint4(WaveGetLaneCount(),lane,
        WaveReadLaneAt(value,31-lane),QuadReadAcrossX(value)));
    Destination.Store4(id.x*32+16,uint4(QuadReadAcrossY(value),WaveActiveSum(value),WaveReadLaneFirst(id.x),id.x));
}
