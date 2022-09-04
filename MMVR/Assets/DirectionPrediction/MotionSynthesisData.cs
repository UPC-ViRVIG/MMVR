using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif
using MotionMatching;

using Joint = MotionMatching.Skeleton.Joint;

[CreateAssetMenu(fileName = "MotionSynthesisData", menuName = "MotionSynthesis/MotionSynthesisData")]
public class MotionSynthesisData : ScriptableObject
{
    public List<TextAsset> BVHs;
    public TextAsset BVHTPose; // BVH with a TPose in the first frame, used for retargeting
    public float UnitScale = 1.0f;
    public float3 HipsForwardLocalVector = new float3(0, 0, 1); // Local vector (axis) pointing in the forward direction of the hips
    public bool SmoothSimulationBone; // Smooth the simulation bone (articial root added during pose extraction) using Savitzky-Golay filter
    public List<JointToMecanim> SkeletonToMecanim = new List<JointToMecanim>();

    private List<BVHAnimation> Animations;
    private PoseSet PoseSet;
    private PoseDataset PoseDataset;
    private TrackersDataset TrackersDataset;

    // Information extracted form T-Pose
    [SerializeField] private float3[] JointsLocalForward; // Local forward vector of each joint 
    public bool JointsLocalForwardError { get { return JointsLocalForward == null; } }

    private void ImportAnimations()
    {
        Animations = new List<BVHAnimation>();
        PROFILE.BEGIN_SAMPLE_PROFILING("BVH Import");
        for (int i = 0; i < BVHs.Count; i++)
        {
            BVHImporter importer = new BVHImporter();
            BVHAnimation animation = importer.Import(BVHs[i], UnitScale);
            Animations.Add(animation);
            // Add Mecanim mapping information
            animation.UpdateMecanimInformation(this);
        }
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("BVH Import");
    }

    public PoseSet GetOrImportPoseSet()
    {
        if (PoseSet == null)
        {
            PROFILE.BEGIN_SAMPLE_PROFILING("Pose Import");
            PoseSerializer serializer = new PoseSerializer();
            if (!serializer.Deserialize(GetAssetPath(), name, null, out PoseSet))
            {
                Debug.LogError("Failed to read pose set. Creating it in runtime instead.");
                ImportPoseSetIfNeeded();
            }
            PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Import");
        }
        return PoseSet;
    }

    private void ImportPoseSetIfNeeded(bool force = false)
    {
        if (PoseSet == null || force)
        {
            ImportAnimations();
            PoseSet = new PoseSet();
            PoseSet.SetSkeletonFromBVH(Animations[0].Skeleton);
            for (int i = 0; i < Animations.Count; i++)
            {
                BVHAnimation animation = Animations[i];
                PoseExtractor poseExtractor = new PoseExtractor();
                if (!poseExtractor.Extract(animation, PoseSet, this))
                {
                    Debug.LogError("[FeatureDebug] Failed to extract pose from BVHAnimation. BVH Index: " + i);
                }
            }
        }
    }

    public PoseDataset GetOrImportPoseDataset()
    {
        if (PoseDataset == null)
        {
            PROFILE.BEGIN_SAMPLE_PROFILING("PoseDataset Import");
            PoseDataset = new PoseDataset();
            if (!PoseDataset.Deserialize(GetAssetPath(), name))
            {
                Debug.LogError("Failed to read pose dataset. Creating it in runtime instead.");
                ImportPoseDataset();
            }
            PROFILE.END_AND_PRINT_SAMPLE_PROFILING("PoseDataset Import");
        }
        return PoseDataset;
    }

    private void ImportPoseDataset()
    {
        ImportPoseSetIfNeeded();
        Debug.Assert(TrackersDataset != null, "TrackersDataset must be imported before PoseDataset");
        PoseDataset = new PoseDataset(PoseSet, TrackersDataset);
    }

    public void GetMeanAndStdTrackersInfo(out float[] mean, out float[] std)
    {
        TrackersDataset.DeserializeMeanAndStd(GetAssetPath(), name, out mean, out std);
    }

    public void GetMeanAndStdPose(out float[] mean, out float[] std)
    {
        PoseDataset.DeserializeMeanAndStd(GetAssetPath(), name, out mean, out std);
    }

    public TrackersDataset GetOrImportTrackersDataset()
    {
        if (TrackersDataset == null)
        {
            PROFILE.BEGIN_SAMPLE_PROFILING("TrackersDataset Import");
            TrackersDataset = new TrackersDataset();
            if (!TrackersDataset.Deserialize(GetAssetPath(), name))
            {
                Debug.LogError("Failed to read trackers dataset. Creating it in runtime instead.");
                ImportTrackersDataset();
            }
            PROFILE.END_AND_PRINT_SAMPLE_PROFILING("TrackersDataset Import");
        }
        return TrackersDataset;
    }

    private void ImportTrackersDataset()
    {
        ImportPoseSetIfNeeded();
        TrackersDataset = new TrackersDataset(PoseSet, this);
    }

    private void ComputeJointsLocalForward()
    {
        // Import T-Pose
        BVHImporter bvhImporter = new BVHImporter();
        BVHAnimation tposeAnimation = bvhImporter.Import(BVHTPose, UnitScale, true);
        JointsLocalForward = new float3[tposeAnimation.Skeleton.Joints.Count + 1]; // +1 for the simulation bone
                                                                                   // Find forward character vector by projecting hips forward vector onto the ground
        Quaternion[] localRotations = tposeAnimation.Frames[0].LocalRotations;
        float3 hipsWorldForwardProjected = math.mul(localRotations[0], HipsForwardLocalVector);
        hipsWorldForwardProjected.y = 0;
        hipsWorldForwardProjected = math.normalize(hipsWorldForwardProjected);
        // Find right character vector by rotating Y-Axis 90 degrees (Unity is Left-Handed and Y-Axis is Up)
        float3 hipsWorldRightProjected = math.mul(quaternion.AxisAngle(math.up(), math.radians(90.0f)), hipsWorldForwardProjected);
        // Compute JointsLocalForward based on the T-Pose
        JointsLocalForward[0] = math.forward();
        for (int i = 1; i < JointsLocalForward.Length; i++)
        {
            quaternion worldRot = quaternion.identity;
            int joint = i - 1;
            while (joint != 0) // while not root
            {
                worldRot = math.mul(localRotations[joint], worldRot);
                joint = tposeAnimation.Skeleton.Joints[joint].ParentIndex;
            }
            worldRot = math.mul(localRotations[0], worldRot); // root
            joint = i - 1;
            // Change to Local
            if (!GetMecanimBone(tposeAnimation.Skeleton.Joints[joint].Name, out HumanBodyBones bone))
            {
                Debug.LogError("[FeatureDebug] Failed to find Mecanim bone for joint " + tposeAnimation.Skeleton.Joints[joint].Name);
            }
            float3 worldForward = hipsWorldForwardProjected;
            if (HumanBodyBonesExtensions.IsLeftArmBone(bone))
            {
                worldForward = -hipsWorldRightProjected;
            }
            else if (HumanBodyBonesExtensions.IsRightArmBone(bone))
            {
                worldForward = hipsWorldRightProjected;
            }
            JointsLocalForward[i] = math.mul(math.inverse(worldRot), worldForward);
        }
    }

    /// <summary>
    /// Returns the local forward vector of the iven joint index (after adding simulation bone)
    /// Vector computed from the T-Pose BVH and HipsForwardLocalVector
    /// </summary>
    public float3 GetLocalForward(int jointIndex)
    {
        Debug.Assert(!JointsLocalForwardError, "JointsLocalForward is not initialized");
        return JointsLocalForward[jointIndex];
    }

    public bool GetMecanimBone(string jointName, out HumanBodyBones bone)
    {
        for (int i = 0; i < SkeletonToMecanim.Count; i++)
        {
            if (SkeletonToMecanim[i].Name == jointName)
            {
                bone = SkeletonToMecanim[i].MecanimBone;
                return true;
            }
        }
        bone = HumanBodyBones.LastBone;
        return false;
    }

    public bool GetJointName(HumanBodyBones bone, out string jointName)
    {
        for (int i = 0; i < SkeletonToMecanim.Count; i++)
        {
            if (SkeletonToMecanim[i].MecanimBone == bone)
            {
                jointName = SkeletonToMecanim[i].Name;
                return true;
            }
        }
        jointName = "";
        return false;
    }

    public string GetAssetPath()
    {
        string assetPath = AssetDatabase.GetAssetPath(this);
        return Path.Combine(Application.dataPath, assetPath.Remove(assetPath.Length - ".asset".Length, 6).Remove(0, "Assets".Length + 1));
    }

    [System.Serializable]
    public struct JointToMecanim
    {
        public string Name;
        public HumanBodyBones MecanimBone;

        public JointToMecanim(string name, HumanBodyBones mecanimBone)
        {
            Name = name;
            MecanimBone = mecanimBone;
        }
    }

    public void GenerateDatabases()
    {
        PROFILE.BEGIN_SAMPLE_PROFILING("Pose Extract");
        ImportPoseSetIfNeeded(true);
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Extract");

        PROFILE.BEGIN_SAMPLE_PROFILING("Pose Serialize");
        PoseSerializer poseSerializer = new PoseSerializer();
        poseSerializer.Serialize(PoseSet, GetAssetPath(), this.name);
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Serialize");

        ComputeJointsLocalForward();

        PROFILE.BEGIN_SAMPLE_PROFILING("TrackersDataset Creation");
        ImportTrackersDataset();
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("TrackersDataset Creation");

        PROFILE.BEGIN_SAMPLE_PROFILING("TrackersDataset Serialize");
        TrackersDataset.Serialize(GetAssetPath(), this.name);
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("TrackersDataset Serialize");

        PROFILE.BEGIN_SAMPLE_PROFILING("PoseDataset Creation");
        ImportPoseDataset();
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("PoseDataset Creation");

        PROFILE.BEGIN_SAMPLE_PROFILING("PoseDataset Serialize");
        PoseDataset.Serialize(GetAssetPath(), this.name);
        PROFILE.END_AND_PRINT_SAMPLE_PROFILING("PoseDataset Serialize");

        AssetDatabase.Refresh();
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(MotionSynthesisData))]
public class MotionSynthesisDataEditor : Editor
{
    private bool SkeletonToMecanimFoldout;
    private bool FeatureSelectorFoldout;

    public override void OnInspectorGUI()
    {
        MotionSynthesisData data = (MotionSynthesisData)target;

        // BVH
        EditorGUILayout.LabelField("BVHs", EditorStyles.boldLabel);
        EditorGUI.indentLevel++;
        for (int i = 0; i < (data.BVHs == null ? 0 : data.BVHs.Count); i++)
        {
            data.BVHs[i] = (TextAsset)EditorGUILayout.ObjectField(data.BVHs[i], typeof(TextAsset), false);
        }
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Add BVH"))
        {
            if (data.BVHs == null) data.BVHs = new List<TextAsset>();
            data.BVHs.Add(null);
        }
        if (GUILayout.Button("Remove BVH"))
        {
            data.BVHs.RemoveAt(data.BVHs.Count - 1);
        }
        EditorGUILayout.EndHorizontal();
        EditorGUI.indentLevel--;
        if (data.BVHs == null) return;
        // BVH TPose
        data.BVHTPose = (TextAsset)EditorGUILayout.ObjectField(new GUIContent("BVH with TPose", "BVH with a TPose in the first frame, used for retargeting"),
                                                               data.BVHTPose, typeof(TextAsset), false);
        // UnitScale
        EditorGUILayout.BeginHorizontal();
        data.UnitScale = EditorGUILayout.FloatField("Unit Scale", data.UnitScale);
        if (GUILayout.Button("m")) data.UnitScale = 1.0f;
        if (GUILayout.Button("cm")) data.UnitScale = 0.01f;
        EditorGUILayout.EndHorizontal();
        // DefaultHipsForward
        data.HipsForwardLocalVector = EditorGUILayout.Vector3Field(new GUIContent("Hips Forward Local Vector", "Local vector (axis) pointing in the forward direction of the hips"),
                                                                   data.HipsForwardLocalVector);
        if (math.abs(math.length(data.HipsForwardLocalVector) - 1.0f) > 1E-6f)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.HelpBox("Hips Forward Local Vector should be normalized", MessageType.Warning);
            if (GUILayout.Button("Fix")) data.HipsForwardLocalVector = math.normalize(data.HipsForwardLocalVector);
            EditorGUILayout.EndHorizontal();
        }

        // SmoothSimulationBone
        data.SmoothSimulationBone = EditorGUILayout.Toggle(new GUIContent("Smooth Simulation Bone", "Smooth the simulation bone (articial root added during pose extraction) using Savitzky-Golay filter"),
                                                           data.SmoothSimulationBone);

        // SkeletonToMecanim
        if (GUILayout.Button("Read Skeleton from BVH"))
        {
            BVHImporter importer = new BVHImporter();
            BVHAnimation animation = importer.Import(data.BVHTPose != null ? data.BVHTPose : data.BVHs[0], data.UnitScale);
            // Check if SkeletonToMecanim should be reset
            bool shouldResetSkeletonToMecanim = true || data.SkeletonToMecanim.Count != animation.Skeleton.Joints.Count;
            if (!shouldResetSkeletonToMecanim)
            {
                foreach (MotionSynthesisData.JointToMecanim jtm in data.SkeletonToMecanim)
                {
                    if (!animation.Skeleton.Find(jtm.Name, out _))
                    {
                        shouldResetSkeletonToMecanim = true;
                        break;
                    }
                }
            }
            if (shouldResetSkeletonToMecanim)
            {
                data.SkeletonToMecanim.Clear();
                foreach (Joint joint in animation.Skeleton.Joints)
                {
                    HumanBodyBones bone;
                    try
                    {
                        bone = (HumanBodyBones)Enum.Parse(typeof(HumanBodyBones), joint.Name);
                    }
                    catch (Exception)
                    {
                        bone = HumanBodyBones.LastBone;
                    }
                    data.SkeletonToMecanim.Add(new MotionSynthesisData.JointToMecanim(joint.Name, bone));
                }
            }
        }

        // Display SkeletonToMecanim
        SkeletonToMecanimFoldout = EditorGUILayout.BeginFoldoutHeaderGroup(SkeletonToMecanimFoldout, "Skeleton to Mecanim");
        if (SkeletonToMecanimFoldout)
        {
            EditorGUI.indentLevel++;
            for (int i = 0; i < data.SkeletonToMecanim.Count; i++)
            {
                MotionSynthesisData.JointToMecanim jtm = data.SkeletonToMecanim[i];
                EditorGUILayout.BeginHorizontal();
                GUI.contentColor = jtm.MecanimBone == HumanBodyBones.LastBone ? new Color(1.0f, 0.6f, 0.6f) : Color.white;
                HumanBodyBones newHumanBodyBone = (HumanBodyBones)EditorGUILayout.EnumPopup(jtm.Name, jtm.MecanimBone);
                GUI.contentColor = Color.white;
                jtm.MecanimBone = newHumanBodyBone;
                data.SkeletonToMecanim[i] = jtm;
                EditorGUILayout.EndHorizontal();
            }
            EditorGUI.indentLevel--;
        }
        EditorGUILayout.EndFoldoutHeaderGroup();

        // Generate Databases
        EditorGUILayout.Separator();
        if (GUILayout.Button("Generate Databases", GUILayout.Height(30)))
        {
            data.GenerateDatabases();
        }

        // Error Check
        if (data.JointsLocalForwardError)
        {
            EditorGUILayout.HelpBox("Internal error detected. Please regenerate databases.", MessageType.Error);
        }

        // Save
        if (GUI.changed)
        {
            EditorUtility.SetDirty(target);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }
    }
}
#endif