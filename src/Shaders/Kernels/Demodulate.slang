RWTexture2D<float4> gImage;
#ifdef gModulate
RWTexture2D<float4> gOutput;
#endif
Texture2D<float4> gAlbedo;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 extent;
    gImage.GetDimensions(extent.x, extent.y);
    if (any(id.xy >= extent)) return;

    const float3 albedo = gAlbedo[id.xy].rgb + 1e-2;
    float4 color  = gImage[id.xy];
    #ifdef gModulate
    color.rgb *= albedo;
    gOutput[id.xy] = color;
	#else
	color.rgb /= albedo;
    gImage[id.xy] = color;
	#endif
}