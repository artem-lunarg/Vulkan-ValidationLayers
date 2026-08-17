/*
 * Copyright (c) 2022-2026 The Khronos Group Inc.
 * Copyright (c) 2022-2026 RasterGrid Kft.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "video_objects.h"

class NegativeSyncValVideo : public VkVideoSyncLayerTest {};

class NewNegativeSyncValVideoSubmitOnly : public VkVideoSyncLayerTest {
  public:
    NewNegativeSyncValVideoSubmitOnly() {
        settings_[0] = {OBJECT_LAYER_NAME, "syncval_record_time_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1,
                        &record_time_validation_};
        settings_[1] = {OBJECT_LAYER_NAME, "syncval_full_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1,
                        &full_validation_};
        settings_info_ = vku::InitStructHelper();
        settings_info_.settingCount = size32(settings_);
        settings_info_.pSettings = settings_.data();

        validation_features_ = vku::InitStructHelper(&settings_info_);
        validation_features_.enabledValidationFeatureCount = size32(enables_);
        validation_features_.pEnabledValidationFeatures = enables_.data();
        validation_features_.disabledValidationFeatureCount = size32(disables_);
        validation_features_.pDisabledValidationFeatures = disables_.data();
        SetInstancePNext(&validation_features_);
    }

  private:
    VkBool32 record_time_validation_ = VK_FALSE;
    VkBool32 full_validation_ = VK_TRUE;
    std::array<VkLayerSettingEXT, 2> settings_{};
    VkLayerSettingsCreateInfoEXT settings_info_{};
    const std::array<VkValidationFeatureEnableEXT, 1> enables_ = {VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
    const std::array<VkValidationFeatureDisableEXT, 4> disables_ = {
        VK_VALIDATION_FEATURE_DISABLE_THREAD_SAFETY_EXT, VK_VALIDATION_FEATURE_DISABLE_API_PARAMETERS_EXT,
        VK_VALIDATION_FEATURE_DISABLE_OBJECT_LIFETIMES_EXT, VK_VALIDATION_FEATURE_DISABLE_CORE_CHECKS_EXT};
    VkValidationFeaturesEXT validation_features_{};
};

TEST_F(NegativeSyncValVideo, DecodeOutputPicture) {
    TEST_DESCRIPTION("Test video decode output picture sync hazard");

    RETURN_IF_SKIP(Init());

    VideoConfig config = GetConfigDecode();
    if (!config) {
        GTEST_SKIP() << "Test requires decode support";
    }

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();

    cb.Begin();
    cb.BeginVideoCoding(context.Begin());
    cb.ControlVideoCoding(context.Control().Reset());

    cb.DecodeVideo(context.DecodeFrame());

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    cb.DecodeVideo(context.DecodeFrame());
    m_errorMonitor->VerifyFound();

    cb.EndVideoCoding(context.End());
    cb.End();
}

TEST_F(NewNegativeSyncValVideoSubmitOnly, DecodeOutputPicture) {
    RETURN_IF_SKIP(Init());

    VideoConfig config = GetConfigDecode();
    if (!config) {
        GTEST_SKIP() << "Test requires decode support";
    }

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();
    cb.Begin();
    cb.BeginVideoCoding(context.Begin());
    cb.ControlVideoCoding(context.Control().Reset());
    cb.DecodeVideo(context.DecodeFrame());
    cb.DecodeVideo(context.DecodeFrame());
    cb.EndVideoCoding(context.End());
    cb.End();

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    context.Queue().Submit(cb);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeSyncValVideo, DecodeReconstructedPicture) {
    TEST_DESCRIPTION("Test video decode reconstructed picture sync hazard");

    RETURN_IF_SKIP(Init());

    auto config = GetConfig(FilterConfigs(GetConfigsWithDpbSlots(GetConfigsWithReferences(GetConfigsDecode()), 2),
                                          [](const VideoConfig& config) { return config.SupportsDecodeOutputDistinct(); }));
    if (!config) {
        GTEST_SKIP() << "Test requires video decode support with references, 2 DPB slots, and distinct mode support";
    }

    config.SessionCreateInfo()->maxDpbSlots = 2;
    config.SessionCreateInfo()->maxActiveReferencePictures = 1;

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();

    cb.Begin();
    cb.BeginVideoCoding(context.Begin().AddResource(0, 0).AddResource(1, 1));
    cb.ControlVideoCoding(context.Control().Reset());

    cb.DecodeVideo(context.DecodeFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.DecodeOutput()->MemoryBarrier());

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    cb.DecodeVideo(context.DecodeReferenceFrame(0));
    m_errorMonitor->VerifyFound();

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    cb.DecodeVideo(context.DecodeFrame(0));
    m_errorMonitor->VerifyFound();

    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(0, 1));
    cb.DecodeVideo(context.DecodeReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(0, 1));
    vk::CmdPipelineBarrier2KHR(cb, context.DecodeOutput()->MemoryBarrier());

    cb.DecodeVideo(context.DecodeFrame(1).AddReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.DecodeOutput()->MemoryBarrier());

    cb.EndVideoCoding(context.End());
    cb.End();
}

TEST_F(NegativeSyncValVideo, DecodeReferencePicture) {
    TEST_DESCRIPTION("Test video decode reference picture sync hazard");

    RETURN_IF_SKIP(Init());

    auto config = GetConfig(GetConfigsWithDpbSlots(GetConfigsWithReferences(GetConfigsDecode()), 3));
    if (!config) {
        GTEST_SKIP() << "Test requires video decode support with references and 3 DPB slots";
    }

    config.SessionCreateInfo()->maxDpbSlots = 3;
    config.SessionCreateInfo()->maxActiveReferencePictures = 1;

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();

    cb.Begin();
    cb.BeginVideoCoding(context.Begin().AddResource(0, 0).AddResource(1, 1).AddResource(2, 2));
    cb.ControlVideoCoding(context.Control().Reset());

    cb.DecodeVideo(context.DecodeReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.DecodeOutput()->MemoryBarrier());

    cb.DecodeVideo(context.DecodeReferenceFrame(1));
    vk::CmdPipelineBarrier2KHR(cb, context.DecodeOutput()->MemoryBarrier());

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-READ-AFTER-WRITE");
    cb.DecodeVideo(context.DecodeFrame(2).AddReferenceFrame(0));
    m_errorMonitor->VerifyFound();

    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(0, 1));
    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(2, 1));
    cb.DecodeVideo(context.DecodeFrame(2).AddReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.DecodeOutput()->MemoryBarrier());
    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(2, 1));

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-READ-AFTER-WRITE");
    cb.DecodeVideo(context.DecodeFrame(2).AddReferenceFrame(1));
    m_errorMonitor->VerifyFound();

    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(1, 2));
    cb.DecodeVideo(context.DecodeFrame(2).AddReferenceFrame(1));

    cb.EndVideoCoding(context.End());
    cb.End();
}

TEST_F(NegativeSyncValVideo, EncodeBitstream) {
    TEST_DESCRIPTION("Test video encode bitstream sync hazard");

    RETURN_IF_SKIP(Init());

    VideoConfig config = GetConfigEncode();
    if (!config) {
        GTEST_SKIP() << "Test requires Encode support";
    }

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();

    cb.Begin();
    cb.BeginVideoCoding(context.Begin());
    cb.ControlVideoCoding(context.Control().Reset());

    cb.EncodeVideo(context.EncodeFrame());

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    cb.EncodeVideo(context.EncodeFrame());
    m_errorMonitor->VerifyFound();

    cb.EndVideoCoding(context.End());
    cb.End();
}

TEST_F(NegativeSyncValVideo, EncodeReconstructedPicture) {
    TEST_DESCRIPTION("Test video encode reconstructed picture sync hazard");

    RETURN_IF_SKIP(Init());

    auto config = GetConfig(GetConfigsWithDpbSlots(GetConfigsWithReferences(GetConfigsEncode()), 2));
    if (!config) {
        GTEST_SKIP() << "Test requires video encode support with references and 2 DPB slots";
    }

    config.SessionCreateInfo()->maxDpbSlots = 2;
    config.SessionCreateInfo()->maxActiveReferencePictures = 1;

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();

    cb.Begin();
    cb.BeginVideoCoding(context.Begin().AddResource(0, 0).AddResource(1, 1));
    cb.ControlVideoCoding(context.Control().Reset());

    cb.EncodeVideo(context.EncodeFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.Bitstream().MemoryBarrier());

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    cb.EncodeVideo(context.EncodeReferenceFrame(0));
    m_errorMonitor->VerifyFound();

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-WRITE");
    cb.EncodeVideo(context.EncodeFrame(0));
    m_errorMonitor->VerifyFound();

    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(0, 1));
    cb.EncodeVideo(context.EncodeReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(0, 1));
    vk::CmdPipelineBarrier2KHR(cb, context.Bitstream().MemoryBarrier());

    cb.EncodeVideo(context.EncodeFrame(1).AddReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.Bitstream().MemoryBarrier());

    cb.EndVideoCoding(context.End());
    cb.End();
}

TEST_F(NegativeSyncValVideo, EncodeReferencePicture) {
    TEST_DESCRIPTION("Test video encode reference picture sync hazard");

    RETURN_IF_SKIP(Init());

    auto config = GetConfig(GetConfigsWithDpbSlots(GetConfigsWithReferences(GetConfigsEncode()), 3));
    if (!config) {
        GTEST_SKIP() << "Test requires video encode support with references and 3 DPB slots";
    }

    config.SessionCreateInfo()->maxDpbSlots = 3;
    config.SessionCreateInfo()->maxActiveReferencePictures = 1;

    VideoContext context(m_device, config);
    context.CreateAndBindSessionMemory();
    context.CreateResources();

    vkt::CommandBuffer& cb = context.CmdBuffer();

    cb.Begin();
    cb.BeginVideoCoding(context.Begin().AddResource(0, 0).AddResource(1, 1).AddResource(2, 2));
    cb.ControlVideoCoding(context.Control().Reset());

    cb.EncodeVideo(context.EncodeReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.Bitstream().MemoryBarrier());

    cb.EncodeVideo(context.EncodeReferenceFrame(1));
    vk::CmdPipelineBarrier2KHR(cb, context.Bitstream().MemoryBarrier());

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-READ-AFTER-WRITE");
    cb.EncodeVideo(context.EncodeFrame(2).AddReferenceFrame(0));
    m_errorMonitor->VerifyFound();

    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(0, 1));
    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(2, 1));
    cb.EncodeVideo(context.EncodeFrame(2).AddReferenceFrame(0));
    vk::CmdPipelineBarrier2KHR(cb, context.Bitstream().MemoryBarrier());
    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(2, 1));

    m_errorMonitor->SetDesiredError("SYNC-HAZARD-READ-AFTER-WRITE");
    cb.EncodeVideo(context.EncodeFrame(2).AddReferenceFrame(1));
    m_errorMonitor->VerifyFound();

    vk::CmdPipelineBarrier2KHR(cb, context.Dpb()->MemoryBarrier(1, 2));
    cb.EncodeVideo(context.EncodeFrame(2).AddReferenceFrame(1));

    cb.EndVideoCoding(context.End());
    cb.End();
}

TEST_F(NegativeSyncValVideo, EncodeQuantizationMap) {
    TEST_DESCRIPTION("Test video encode quantization map sync hazard");

    AddRequiredExtensions(VK_KHR_VIDEO_ENCODE_QUANTIZATION_MAP_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::videoEncodeQuantizationMap);
    RETURN_IF_SKIP(Init());

    VideoConfig delta_config = GetConfigWithQuantDeltaMap(GetConfigsEncode());
    VideoConfig emphasis_config = GetConfigWithEmphasisMap(GetConfigsEncode());

    if ((!delta_config || (QueueFamilyFlags(delta_config.QueueFamilyIndex()) & VK_QUEUE_TRANSFER_BIT) == 0) &&
        (!emphasis_config || (QueueFamilyFlags(emphasis_config.QueueFamilyIndex()) & VK_QUEUE_TRANSFER_BIT) == 0)) {
        GTEST_SKIP() << "Test case requires video encode queue to also support transfer";
    }

    struct TestConfig {
        VideoConfig config;
        VkVideoEncodeFlagBitsKHR encode_flag;
        VkVideoSessionCreateFlagBitsKHR session_create_flag;
        const VkVideoFormatPropertiesKHR* map_props;
    };

    std::vector<TestConfig> tests;
    if (delta_config) {
        tests.emplace_back(TestConfig{delta_config, VK_VIDEO_ENCODE_WITH_QUANTIZATION_DELTA_MAP_BIT_KHR,
                                      VK_VIDEO_SESSION_CREATE_ALLOW_ENCODE_QUANTIZATION_DELTA_MAP_BIT_KHR,
                                      delta_config.QuantDeltaMapProps()});
    }
    if (emphasis_config) {
        tests.emplace_back(TestConfig{emphasis_config, VK_VIDEO_ENCODE_WITH_EMPHASIS_MAP_BIT_KHR,
                                      VK_VIDEO_SESSION_CREATE_ALLOW_ENCODE_EMPHASIS_MAP_BIT_KHR,
                                      emphasis_config.EmphasisMapProps()});
    }

    for (auto& [config, encode_flag, session_create_flag, map_props] : tests) {
        config.SessionCreateInfo()->flags |= session_create_flag;
        VideoContext context(m_device, config);
        context.CreateAndBindSessionMemory();
        context.CreateResources();

        const auto texel_size = config.GetQuantMapTexelSize(*map_props);
        auto params = context.CreateSessionParamsWithQuantMapTexelSize(texel_size);

        vkt::CommandBuffer& cb = context.CmdBuffer();

        VideoEncodeQuantizationMap quantization_map(m_device, config, *map_props);

        cb.Begin();

        VkClearColorValue clear_value{};
        VkImageSubresourceRange subres_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk::CmdClearColorImage(cb, quantization_map.Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_value, 1, &subres_range);

        cb.BeginVideoCoding(context.Begin().SetSessionParams(params));
        cb.ControlVideoCoding(context.Control().Reset());

        m_errorMonitor->SetDesiredError("SYNC-HAZARD-READ-AFTER-WRITE");
        cb.EncodeVideo(context.EncodeFrame().QuantizationMap(encode_flag, texel_size, quantization_map));
        m_errorMonitor->VerifyFound();

        cb.EndVideoCoding(context.End());
        cb.End();

        cb.Begin();
        cb.BeginVideoCoding(context.Begin().SetSessionParams(params));
        cb.ControlVideoCoding(context.Control().Reset());

        cb.EncodeVideo(context.EncodeFrame().QuantizationMap(encode_flag, texel_size, quantization_map));
        m_errorMonitor->VerifyFound();

        cb.EndVideoCoding(context.End());

        m_errorMonitor->SetDesiredError("SYNC-HAZARD-WRITE-AFTER-READ");
        vk::CmdClearColorImage(cb, quantization_map.Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_value, 1, &subres_range);
        m_errorMonitor->VerifyFound();

        cb.End();
    }

    if (!delta_config || (QueueFamilyFlags(delta_config.QueueFamilyIndex()) & VK_QUEUE_TRANSFER_BIT) == 0 || !emphasis_config ||
        (QueueFamilyFlags(emphasis_config.QueueFamilyIndex()) & VK_QUEUE_TRANSFER_BIT) == 0) {
        GTEST_SKIP() << "Not all quantization map types could be tested";
    }
}
