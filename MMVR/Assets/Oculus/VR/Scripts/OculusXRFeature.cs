/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

#if USING_XR_SDK_OPENXR

using System;
using System.Runtime.InteropServices;
using AOT;
using UnityEngine;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.XR.OpenXR;
using UnityEditor.XR.OpenXR.Features;
#endif

namespace Oculus.XR
{
#if UNITY_EDITOR
	public class OculusXRFeatureEditorConfig
	{
		public const string OpenXrExtensionList =
			"XR_KHR_vulkan_enable " +
			"XR_KHR_D3D11_enable " +
			"XR_OCULUS_common_reference_spaces " +
			"XR_FB_display_refresh_rate " +
			"XR_EXT_performance_settings " +
			"XR_FB_composition_layer_image_layout " +
			"XR_KHR_android_surface_swapchain " +
			"XR_FB_android_surface_swapchain_create " +
			"XR_KHR_composition_layer_color_scale_bias " +
			"XR_FB_color_space " +
			"XR_EXT_hand_tracking " +
			"XR_FB_swapchain_update_state " +
			"XR_FB_swapchain_update_state_opengl_es " +
			"XR_FB_swapchain_update_state_vulkan " +
			"XR_FB_foveation " +
			"XR_FB_foveation_configuration " +
			"XR_FB_foveation_vulkan " +
			"XR_FB_composition_layer_alpha_blend " +
			"XR_KHR_composition_layer_depth " +
			"XR_KHR_composition_layer_cylinder " +
			"XR_KHR_composition_layer_cube " +
			"XR_KHR_composition_layer_equirect2 " +
			"XR_KHR_convert_timespec_time " +
			"XR_KHR_visibility_mask " +
			"XR_FB_render_model " +
			"XR_FB_spatial_entity " +
			"XR_FB_spatial_entity_query " +
			"XR_FB_spatial_entity_storage " +
			"XR_FB_keyboard_tracking " +
			"XR_FB_passthrough " +
			"XR_FB_triangle_mesh " +
			"XR_FB_passthrough_keyboard_hands " +
			"XR_OCULUS_audio_device_guid " +
			"XR_FB_common_events " +
			"XR_FB_space_warp " +
			"XR_FB_hand_tracking_capsules " +
			"XR_FB_hand_tracking_mesh " +
			"XR_FB_hand_tracking_aim " +
			""
			;
	}
#endif

	/// <summary>
	/// OculusXR Feature for OpenXR
	/// </summary>
#if UNITY_EDITOR
	[OpenXRFeature(UiName = "OculusXR Feature",
		BuildTargetGroups = new[] { BuildTargetGroup.Standalone, BuildTargetGroup.Android },
		Company = "Oculus",
		Desc = "OculusXR Feature for OpenXR.",
		DocumentationLink = "https://developer.oculus.com/",
		OpenxrExtensionStrings = OculusXRFeatureEditorConfig.OpenXrExtensionList,
		Version = "0.0.1",
		FeatureId = featureId)]
#endif
	public class OculusXRFeature : OpenXRFeature
	{
		/// <summary>
		/// The feature id string. This is used to give the feature a well known id for reference.
		/// </summary>
		public const string featureId = "com.oculus.openxr.feature.oculusxr";

		/// <inheritdoc />
		protected override IntPtr HookGetInstanceProcAddr(IntPtr func)
		{
			OVRPlugin.UnityOpenXR.Enabled = true;

			Debug.Log($"[OculusXRFeature] HookGetInstanceProcAddr: {func}");

			Debug.Log($"[OculusXRFeature] SetClientVersion");
			OVRPlugin.UnityOpenXR.SetClientVersion();

			return OVRPlugin.UnityOpenXR.HookGetInstanceProcAddr(func);
		}

		/// <inheritdoc />
		protected override bool OnInstanceCreate(ulong xrInstance)
		{
			// here's one way you can grab the instance
			Debug.Log($"[OculusXRFeature] OnInstanceCreate: {xrInstance}");
			return OVRPlugin.UnityOpenXR.OnInstanceCreate(xrInstance);
		}

		/// <inheritdoc />
		protected override void OnInstanceDestroy(ulong xrInstance)
		{
			// here's one way you can grab the instance
			Debug.Log($"[OculusXRFeature] OnInstanceDestroy: {xrInstance}");
			OVRPlugin.UnityOpenXR.OnInstanceDestroy(xrInstance);
		}

		/// <inheritdoc />
		protected override void OnSessionCreate(ulong xrSession)
		{
			// here's one way you can grab the session
			Debug.Log($"[OculusXRFeature] OnSessionCreate: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionCreate(xrSession);
		}

		/// <inheritdoc />
		protected override void OnAppSpaceChange(ulong xrSpace)
		{
			Debug.Log($"[OculusXRFeature] OnAppSpaceChange: {xrSpace}");
			OVRPlugin.UnityOpenXR.OnAppSpaceChange(xrSpace);
		}

		/// <inheritdoc />
		protected override void OnSessionStateChange(int oldState, int newState)
		{
			Debug.Log($"[OculusXRFeature] OnSessionStateChange: {oldState} -> {newState}");
			OVRPlugin.UnityOpenXR.OnSessionStateChange(oldState, newState);
		}

		/// <inheritdoc />
		protected override void OnSessionBegin(ulong xrSession)
		{
			Debug.Log($"[OculusXRFeature] OnSessionBegin: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionBegin(xrSession);
		}

		/// <inheritdoc />
		protected override void OnSessionEnd(ulong xrSession)
		{
			Debug.Log($"[OculusXRFeature] OnSessionEnd: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionEnd(xrSession);
		}

		/// <inheritdoc />
		protected override void OnSessionExiting(ulong xrSession)
		{
			Debug.Log($"[OculusXRFeature] OnSessionExiting: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionExiting(xrSession);
		}

		/// <inheritdoc />
		protected override void OnSessionDestroy(ulong xrSession)
		{
			Debug.Log($"[OculusXRFeature] OnSessionDestroy: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionDestroy(xrSession);
		}

		// protected override void OnSessionLossPending(ulong xrSession) {}
		// protected override void OnInstanceLossPending (ulong xrInstance) {}
		// protected override void OnSystemChange(ulong xrSystem) {}
		// protected override void OnFormFactorChange (int xrFormFactor) {}
		// protected override void OnViewConfigurationTypeChange (int xrViewConfigurationType) {}
		// protected override void OnEnvironmentBlendModeChange (int xrEnvironmentBlendMode) {}
		// protected override void OnEnabledChange() {}
	}
}

#endif
