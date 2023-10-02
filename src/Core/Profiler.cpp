#include "Profiler.hpp"
#include "CommandBuffer.hpp"

#include <imgui/imgui.h>

namespace ptvk {

std::shared_ptr<Profiler::ProfilerSample> Profiler::mCurrentSample;
std::vector<std::shared_ptr<Profiler::ProfilerSample>> Profiler::mSampleHistory;
uint32_t Profiler::mSampleHistoryCount = 0;
std::optional<std::chrono::high_resolution_clock::time_point> Profiler::mFrameStart = std::nullopt;
std::deque<float> Profiler::mFrameTimes;
uint32_t Profiler::mFrameTimeCount = 32;

inline std::optional<std::pair<ImVec2,ImVec2>> DrawTimelineSample(const Profiler::ProfilerSample& s, const float t0, const float t1, const float x_min, const float x_max, const float y, const float height) {
	const ImVec2 p_min = ImVec2(x_min + t0*(x_max - x_min), y);
	const ImVec2 p_max = ImVec2(x_min + t1*(x_max - x_min), y + height);
	if (p_max.x < x_min || p_max.x > x_max) return {};

	const ImVec2 mousePos = ImGui::GetMousePos();
	bool hovered = (mousePos.y > p_min.y && mousePos.y < p_max.y && mousePos.x > p_min.x && mousePos.x < p_max.x);
	if (hovered) {
		ImGui::BeginTooltip();
		const std::string label = s.mLabel + " (" + std::to_string(std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(s.mDuration).count()) + "ms)";
		ImGui::Text("%s", label.c_str());
		ImGui::EndTooltip();
	}

	ImGui::GetWindowDrawList()->AddRectFilled(p_min, p_max, ImGui::GetColorU32(hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button), 4);

	const ImVec4 clipRect = ImVec4(p_min.x, p_min.y, p_max.x, p_max.y);
	ImGui::GetWindowDrawList()->AddText(nullptr, 0, p_min, ImGui::GetColorU32(ImGuiCol_Text), s.mLabel.c_str(), nullptr, 0, &clipRect);
	return std::make_pair(p_min, p_max);
}

void Profiler::DrawTimeline() {
	std::chrono::high_resolution_clock::time_point t_min = mSampleHistory[0]->mStartTime;
	std::chrono::high_resolution_clock::time_point t_max = t_min;
	for (const auto& f : mSampleHistory) {
		if (f->mStartTime < t_min) t_min = f->mStartTime;
		if (auto t = f->mStartTime + f->mDuration; t > t_max) t_max = t;
	}

	const float inv_dt = 1/std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(t_max - t_min).count();

	const ImVec2 w_min = ImVec2(ImGui::GetWindowContentRegionMin().x + ImGui::GetWindowPos().x, ImGui::GetWindowContentRegionMin().y + ImGui::GetWindowPos().y);
	const float x_max = w_min.x + ImGui::GetWindowContentRegionWidth();

	float height = 18;
	float header_height = 24;
	float pad = 4;

	float y_min = w_min.y;

	// profiler sample history
	{
		const ImVec4 clipRect = ImVec4(w_min.x, y_min, x_max, y_min + header_height);
		ImGui::GetWindowDrawList()->AddText(nullptr, 0, ImVec2(w_min.x, y_min), ImGui::GetColorU32(ImGuiCol_Text), "CPU Profiler Samples");
		y_min += header_height;

		std::stack<std::pair<std::shared_ptr<Profiler::ProfilerSample>, float>> todo;
		for (const auto& f : mSampleHistory) todo.push(std::make_pair(f, 0.f));
		while (!todo.empty()) {
			auto[s,l] = todo.top();
			todo.pop();

			const float t0 = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(s->mStartTime - t_min).count() * inv_dt;
			const float t1 = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(s->mStartTime - t_min + s->mDuration).count() * inv_dt;
			const float h = height;
			auto r = DrawTimelineSample(*s, t0, t1, w_min.x, x_max, y_min + l, h);
			if (!r) continue;

			const auto[p_min,p_max] = *r;

			for (const auto& c : s->mChildren)
				todo.push(std::make_pair(c, l + h + pad));
		}
	}
}

void Profiler::DrawFrameTimeGraph() {
	float fps_timer = 0;
	uint32_t fps_counter = 0;
	std::vector<float> frame_times(mFrameTimes.size());
	for (uint32_t i = 0; i < mFrameTimes.size(); i++) {
		if (fps_timer < 1000.f) {
			fps_timer += mFrameTimes[i];
			fps_counter++;
		}
		frame_times[i] = mFrameTimes[i];
	}

	ImGui::Text("%.1f fps (%.1f ms)", fps_counter/(fps_timer/1000), fps_timer/fps_counter);
	ImGui::SliderInt("Frame Time Count", reinterpret_cast<int*>(&mFrameTimeCount), 2, 256);
	if (frame_times.size() > 1) ImGui::PlotLines("Frame Times", frame_times.data(), (uint32_t)frame_times.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 64));
}


ProfilerScope::ProfilerScope(const std::string& label, const CommandBuffer* cmd, const float4& color) : mCommandBuffer(cmd) {
	Profiler::BeginSample(label, color);
	if (mCommandBuffer) {
		vk::DebugUtilsLabelEXT info = {};
		std::copy_n(&color.x, 4, info.color.data());
		info.pLabelName = label.c_str();
		(*mCommandBuffer)->beginDebugUtilsLabelEXT(info);
	}
}
ProfilerScope::~ProfilerScope() {
	if (mCommandBuffer)
		(*mCommandBuffer)->endDebugUtilsLabelEXT();
	Profiler::EndSample();
}

}