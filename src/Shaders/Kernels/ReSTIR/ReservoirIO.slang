#include "PathGeneration.slang"

RWByteAddressBuffer gPathReservoirs[2];
#define gPathReservoirsIn gPathReservoirs[gReservoirIndex]
#define gPathReservoirsOut gPathReservoirs[gReservoirIndex ^ 1]

uint FlattenIndex(const uint2 id) {
    return id.y * gOutputSize.x + id.x;
}

uint PreIncrement(inout uint address, const uint stride, const uint index, const uint count) {
    uint val = address;
    address += count * stride;
    return val + index * stride;
}
PathReservoir LoadReservoir(RWByteAddressBuffer buf, const uint2 id) {
    const uint index = FlattenIndex(id);
    const uint count = gOutputSize.x * gOutputSize.y;
    if (index >= count) return PathReservoir();

    PathReservoir r = PathReservoir();

    uint address = 0;
    float2 data = buf.Load<float2>(PreIncrement(address, 8, index, count));
    r.mW = data[0];
    r.mM = data[1];

    if (r.mW > 0 && r.mM > 0)
        r.mSample = buf.Load<PathSample>(PreIncrement(address, PathSampleSize, index, count));
    else
        r.mSample = PathSample();

    return r;
}
void StoreReservoir(RWByteAddressBuffer buf, const uint2 id, const PathReservoir r) {
    const uint index = FlattenIndex(id);
    const uint count = gOutputSize.x * gOutputSize.y;
    if (index >= count) return;

    uint address = 0;
    buf.Store<float2>(PreIncrement(address, 8, index, count), float2(r.mW, r.mM));

    if (r.mW > 0 && r.mM > 0)
        buf.Store<PathSample>(PreIncrement(address, PathSampleSize, index, count), r.mSample);
}