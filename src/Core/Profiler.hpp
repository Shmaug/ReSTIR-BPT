#pragma once

#include <list>
#include "Utils.hpp"
#include <Common/Common.h>

namespace ptvk {

class Profiler {
public:
	inline static void BeginSample(const std::string& label, const float4& color = float4(1)) {
		auto s = std::make_shared<ProfilerSample>(mCurrentSample, label, color);
		if (mCurrentSample)
			mCurrentSample = mCurrentSample->mChildren.emplace_back(s);
		else
			mCurrentSample = s;
	}
	inline static void EndSample() {
		if (!mCurrentSample) throw std::logic_error("cannot call end_sample without first calling begin_sample");
		mCurrentSample->mDuration += std::chrono::high_resolution_clock::now() - mCurrentSample->mStartTime;
		if (!mCurrentSample->mParent && mSampleHistory.size() < mSampleHistoryCount)
			mSampleHistory.emplace_back(mCurrentSample);
		mCurrentSample = mCurrentSample->mParent;
	}

	inline static void BeginFrame() {
		auto rn = std::chrono::high_resolution_clock::now();
		if (mFrameStart && mFrameTimeCount > 0) {
			auto duration = rn - *mFrameStart;
			mFrameTimes.emplace_back(std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(duration).count());
			while (mFrameTimes.size() > mFrameTimeCount) mFrameTimes.pop_front();
		}
		mFrameStart = rn;
	}

	inline static bool HasHistory() { return !mSampleHistory.empty(); }
	inline static void ResetHistory(uint32_t n) {
		mSampleHistoryCount = n;
		mSampleHistory.clear();
	}

	static void DrawFrameTimeGraph();
	static void DrawTimeline();

	struct ProfilerSample {
		std::shared_ptr<ProfilerSample> mParent;
		std::list<std::shared_ptr<ProfilerSample>> mChildren;
		std::chrono::high_resolution_clock::time_point mStartTime;
		std::chrono::nanoseconds mDuration;
		float4 mColor;
		std::string mLabel;

		ProfilerSample() = default;
		ProfilerSample(const ProfilerSample& s) = default;
		ProfilerSample(ProfilerSample&& s) = default;
		inline ProfilerSample(const std::shared_ptr<ProfilerSample>& parent, const std::string& label, const float4& color)
			: mParent(parent), mColor(color), mLabel(label), mStartTime(std::chrono::high_resolution_clock::now()), mDuration(std::chrono::nanoseconds::zero()) {}
	};

private:
	static std::shared_ptr<ProfilerSample> mCurrentSample;
	static std::vector<std::shared_ptr<ProfilerSample>> mSampleHistory;
	static uint32_t mSampleHistoryCount;
	static std::optional<std::chrono::high_resolution_clock::time_point> mFrameStart;
	static std::deque<float> mFrameTimes;
	static uint32_t mFrameTimeCount;
};

class CommandBuffer;

class ProfilerScope {
private:
	const CommandBuffer* mCommandBuffer;

public:
	ProfilerScope(const std::string& label, const CommandBuffer* cmd = nullptr, const float4& color = float4(1));
	~ProfilerScope();
};

}