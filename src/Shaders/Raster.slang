#define NO_SCENE_ACCELERATION_STRUCTURE
#include "Scene.slang"

struct PushConstants {
    uint mViewIndex;
    uint mInstanceIndex;
    uint mMaterialAddress;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct Params {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<float4x4> mViewTransforms;
	StructuredBuffer<float4x4> mViewInverseTransforms;
};
ParameterBlock<Params> gParams;

struct VSOut {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint primId : TEXCOORD1;
};

[shader("vertex")]
VSOut vsmain(uint vertexID: SV_VertexID) {
    const MeshInstance instance = reinterpret<MeshInstance>(gScene.mInstances[gPushConstants.mInstanceIndex]);

    const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.VertexInfoIndex()];
    const uint index = LoadTriangleIndicesUniform(vertexInfo, vertexID / 3)[vertexID % 3];
    const float3 vertex = LoadVertexAttribute<float3>(gScene.mVertexBuffers[vertexInfo.GetPositionBuffer()], vertexInfo.GetPositionOffset(), vertexInfo.GetPositionStride(), index);
    const float3 normal = LoadVertexAttribute<float3>(gScene.mVertexBuffers[vertexInfo.GetNormalBuffer()]  , vertexInfo.GetNormalOffset()  , vertexInfo.GetNormalStride()  , index);
    const float2 uv     = LoadVertexAttribute<float2>(gScene.mVertexBuffers[vertexInfo.GetTexcoordBuffer()], vertexInfo.GetTexcoordOffset(), vertexInfo.GetTexcoordStride(), index);

    const float4x4 objectToCamera = mul(gParams.mViewInverseTransforms[gPushConstants.mViewIndex], gScene.mInstanceTransforms[gPushConstants.mInstanceIndex]);

    VSOut o;
    o.position = mul(gParams.mViews[gPushConstants.mViewIndex].mProjection, TransformPoint(objectToCamera, vertex));
    o.position.y = -o.position.y;
    o.normal = normal;
    o.uv = uv;
    o.primId = vertexID / 3;
    return o;
}

[shader("fragment")]
void fsmain(VSOut i, out float4 outputColor: SV_Target0, out uint2 visibility: SV_Target1) {
    const Material material = LoadMaterialUniform(gPushConstants.mMaterialAddress, i.uv);

	#ifdef gUseAlphaMask
    if (alphaMask < gImageCount) {
        if (gScene.mImage4s[material.mImages.GetBaseColor()].SampleLevel(gScene.mStaticSampler, i.uv, 0) < material.AlphaCutoff())
			discard;
	}
	#endif

    outputColor = float4(material.BaseColor() + material.Emission(), 1);

    VisibilityData vis;
    vis.mInstancePrimitiveIndex = (gPushConstants.mInstanceIndex & 0xFFFF) | (i.primId << 16);
    vis.mPackedNormal = PackNormal(i.normal);
    visibility = reinterpret<uint2>(vis);
}