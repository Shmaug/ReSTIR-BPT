#pragma once

#include "PackedTypes.h"

PTVK_NAMESPACE_BEGIN
#ifdef __cplusplus
#pragma pack(push)
#pragma pack(1)
#endif

enum class MaterialParameters {
	eMetallic = 0,
	eRoughness,
	eSubsurface,
	eSheen,
	eSheenTint,
	eSpecular,
	eSpecularTint,
	eAnisotropic,
	eClearcoat,
	eClearcoatGloss,
	eTransmission,
	eEta,
	eAlphaCutoff,
	eBumpScale,
	eNumMaterialParameters
};

// 24 bytes
struct PackedMaterialParameters {
    PackedUnorm16 mPackedData;
	PackedColors mColors;

	float3 BaseColor()     { return mColors.GetColor(); }
    float3 Emission()      { return mColors.GetColorHDR(); }
	float Metallic()       { return mPackedData.Get((uint)MaterialParameters::eMetallic      ); }
	float Roughness()      { return mPackedData.Get((uint)MaterialParameters::eRoughness     ); }
	float Subsurface()     { return mPackedData.Get((uint)MaterialParameters::eSubsurface    ); }
    float Sheen()          { return mPackedData.Get((uint)MaterialParameters::eSheen         ); }
    float SheenTint()      { return mPackedData.Get((uint)MaterialParameters::eSheenTint     ); }
    float Specular()       { return mPackedData.Get((uint)MaterialParameters::eSpecular      ); }
    float SpecularTint()   { return mPackedData.Get((uint)MaterialParameters::eSpecularTint  ); }
	float Anisotropic()    { return mPackedData.Get((uint)MaterialParameters::eAnisotropic   )*2-1; }
	float Clearcoat()      { return mPackedData.Get((uint)MaterialParameters::eClearcoat     ); }
	float ClearcoatGloss() { return mPackedData.Get((uint)MaterialParameters::eClearcoatGloss); }
	float Transmission()   { return mPackedData.Get((uint)MaterialParameters::eTransmission  ); }
	float Eta()            { return mPackedData.Get((uint)MaterialParameters::eEta           )*2; }
	float AlphaCutoff()    { return mPackedData.Get((uint)MaterialParameters::eAlphaCutoff   ); }
	float BumpScale()      { return mPackedData.Get((uint)MaterialParameters::eBumpScale     )*8; }

    SLANG_MUTATING void BaseColor     (const float3 newValue) { mColors.SetColor(newValue); }
    SLANG_MUTATING void Emission      (const float3 newValue) { mColors.SetColorHDR(newValue); }
	SLANG_MUTATING void Metallic      (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eMetallic      , newValue); }
	SLANG_MUTATING void Roughness     (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eRoughness     , newValue); }
	SLANG_MUTATING void Subsurface    (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eSubsurface    , newValue); }
	SLANG_MUTATING void Sheen         (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eSheen         , newValue); }
    SLANG_MUTATING void SheenTint     (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eSheenTint     , newValue); }
	SLANG_MUTATING void Specular      (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eSpecular      , newValue); }
    SLANG_MUTATING void SpecularTint  (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eSpecularTint  , newValue); }
	SLANG_MUTATING void Anisotropic   (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eAnisotropic   , newValue*.5+.5); }
	SLANG_MUTATING void Clearcoat     (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eClearcoat     , newValue); }
	SLANG_MUTATING void ClearcoatGloss(const float newValue)  { mPackedData.Set((uint)MaterialParameters::eClearcoatGloss, newValue); }
	SLANG_MUTATING void Transmission  (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eTransmission  , newValue); }
	SLANG_MUTATING void Eta           (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eEta           , newValue*.5); }
	SLANG_MUTATING void AlphaCutoff   (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eAlphaCutoff   , newValue); }
	SLANG_MUTATING void BumpScale     (const float newValue)  { mPackedData.Set((uint)MaterialParameters::eBumpScale     , newValue/8.0); }
};
#ifdef __cplusplus
static_assert(sizeof(PackedMaterialParameters) == 24);
#elif defined(__SLANG_COMPILER__)
static const uint PackedMaterialParametersSize = sizeof(PackedMaterialParameters);
#endif

// 32 bytes
struct GpuMaterial {
	PackedMaterialParameters mParameters;
	uint2 mImageBits;

	uint GetBaseColorImage()    { return BF_GET(mImageBits[0],  0, 16); }
	uint GetEmissionImage()     { return BF_GET(mImageBits[0], 16, 16); }
	uint GetBumpImage()         { return BF_GET(mImageBits[1],  0, 15); }
	bool GetIsBumpTwoChannel()  { return (bool)BF_GET(mImageBits[1], 15,  1); }
	uint GetPackedParamsImage() { return BF_GET(mImageBits[1], 16, 16); }

	SLANG_MUTATING void SetBaseColorImage(uint newValue)    { BF_SET(mImageBits[0], newValue,  0, 16); }
	SLANG_MUTATING void SetEmissionImage(uint newValue)     { BF_SET(mImageBits[0], newValue, 16, 16); }
	SLANG_MUTATING void SetBumpImage(uint newValue)         { BF_SET(mImageBits[1], newValue,  0, 15); }
	SLANG_MUTATING void SetIsBumpTwoChannel(bool newValue)  { BF_SET(mImageBits[1], (uint)newValue, 15,  1); }
	SLANG_MUTATING void SetPackedParamsImage(uint newValue) { BF_SET(mImageBits[1], newValue, 16, 16); }
};
#ifdef __cplusplus
static_assert(sizeof(GpuMaterial) == 32);
#elif defined(__SLANG_COMPILER__)
static const uint GpuMaterialSize = sizeof(GpuMaterial);
#endif

#ifdef __cplusplus
#pragma pack(pop)
#endif
PTVK_NAMESPACE_END