#pragma once

#include "PackedTypes.h"

PTVK_NAMESPACE_BEGIN

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
	eNumMaterialParameters
};

// 8 bytes
struct MaterialImageIndices {
	uint2 mBits;

	uint GetBaseColor()              { return BF_GET(mBits[0],            0, 16); }
	void SetBaseColor(uint newValue) {        BF_SET(mBits[0], newValue,  0, 16); }

	uint GetEmission()               { return BF_GET(mBits[0],           16, 16); }
	void SetEmission(uint newValue)  {        BF_SET(mBits[0], newValue, 16, 16); }

	uint GetPacked0()                { return BF_GET(mBits[1],            0, 16); }
	void SetPacked0(uint newValue)   {        BF_SET(mBits[1], newValue,  0, 16); }

	uint GetPacked1()                { return BF_GET(mBits[1],           16, 16); }
	void SetPacked1(uint newValue)   {        BF_SET(mBits[1], newValue, 16, 16); }
};

// 32 bytes
struct Material {
	PackedColors mColors;
	MaterialImageIndices mImages;
    PackedUnorm16 mPackedData;

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
};

PTVK_NAMESPACE_END