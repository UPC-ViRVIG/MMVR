using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MotionMatching;
using Unity.Mathematics;
using Unity.Barracuda;

public class VRDirectionPredictor : MonoBehaviour
{
    [Header("Input Devices")]
    public Transform HMDDevice;
    public Transform HMDBodyCenter;
    public Transform LeftController;
    public Transform RightController;

    [Header("Settings")]
    public int FixedFramerate = 90;
    public MotionSynthesisData GroundTruth;
    public NNModel DirectionPredictorSource;
    public Color PredictColor = new Color(1.0f, 0.0f, 0.5f, 1.0f);

    [Header("Debug")]
    public bool ShowArrowIndicator = true;
    public GameObject ArrowIndicatorPrefab;
    public Vector3 ArrowIndicatorDisplacement;

    private PoseDataset PoseDataset;
    public TrackersDataset TrackersDataset { get; private set; }

    public Transform SimulationBone { get; private set; }
    private quaternion CurrentProjHMDRotation;
    private Transform[] TrackerTransforms;
    private float[] CurrentTrackersInfo;
    private float[] DirectionMLPInput;
    public float[] PredictedDirection { get; private set; }

    private float3[][] PreviousVelocities;
    private float3[][] PreviousAngularVelocities;
    private float3[] PreviousPositions;
    private quaternion[] PreviousRotations;
    private const int NumberPastFramesVelocities = 5;

    private GameObject ArrowDirectionIndicator;

    private Direction_MLP DirectionMLP;

    private void Awake()
    {
        TrackersDataset = GroundTruth.GetOrImportTrackersDataset();
        PoseDataset = GroundTruth.GetOrImportPoseDataset();
        CurrentTrackersInfo = new float[TrackersDataset.NumberFeatures];
        DirectionMLPInput = new float[TrackersDataset.NumberFeatures + PoseDataset.JointRotDimension]; // Trackers + Rot
        PredictedDirection = new float[PoseDataset.JointRotDimension]; // Rot
        float3x2 identity = MathExtensions.QuaternionToContinuous(quaternion.identity);
        PredictedDirection[0] = identity.c0[0];
        PredictedDirection[1] = identity.c0[1];
        PredictedDirection[2] = identity.c0[2];
        PredictedDirection[3] = identity.c1[0];
        PredictedDirection[4] = identity.c1[1];
        PredictedDirection[5] = identity.c1[2];
        CurrentProjHMDRotation = quaternion.identity;

        // Previous
        PreviousVelocities = new float3[TrackersDataset.NumberTrackers][];
        for (int i = 0; i < PreviousVelocities.Length; ++i) { PreviousVelocities[i] = new float3[NumberPastFramesVelocities]; }
        PreviousAngularVelocities = new float3[TrackersDataset.NumberTrackers][];
        for (int i = 0; i < PreviousAngularVelocities.Length; ++i) { PreviousAngularVelocities[i] = new float3[NumberPastFramesVelocities]; }
        PreviousPositions = new float3[TrackersDataset.NumberTrackers];
        PreviousRotations = new quaternion[TrackersDataset.NumberTrackers];
        for (int i = 0; i < PreviousRotations.Length; ++i) { PreviousRotations[i] = quaternion.identity; }

        // SimulationBone
        SimulationBone = (new GameObject("SimulationBone")).transform;
        SimulationBone.SetParent(transform, false);
        ArrowDirectionIndicator = Instantiate(ArrowIndicatorPrefab);
        ArrowDirectionIndicator.name = "DirectionIndicator";
        //ArrowDirectionIndicator.transform.SetParent(SimulationBone, false);
        //ArrowDirectionIndicator.transform.localPosition = new float3(0.0f, 0.0f, 0.0f);

        // Trackers
        TrackerTransforms = new Transform[TrackersDataset.NumberTrackers];
        for (int t = 0; t < TrackersDataset.NumberTrackers; t++)
        {
            TrackerTransforms[t] = (new GameObject()).transform;
            string name;
            if (t == 0) name = "HMD";
            else if (t == 1) name = "Left Controller";
            else name = "Right Controller";
            TrackerTransforms[t].name = name;
            TrackerTransforms[t].SetParent(transform, false);
        }


        // Neural Networks
        if (DirectionPredictorSource != null)
        {
            // Input Size: Trackers + Simulation Bone Rotation
            DirectionMLP = new Direction_MLP(DirectionPredictorSource, TrackersDataset.NumberFeatures + PoseDataset.JointRotDimension);
        }
    }

    public void SetEnabledDebug(bool enabled)
    {
        bool newEnabled = ShowArrowIndicator && enabled;
        if (ArrowDirectionIndicator.activeSelf != newEnabled) ArrowDirectionIndicator.SetActive(newEnabled);
    }

    public void SetPositionDebug(Vector3 pos)
    {
        ArrowDirectionIndicator.transform.position = pos + ArrowIndicatorDisplacement;
    }

    public quaternion GetPredictedRotation()
    {
        TryUpdate();
        return SimulationBone.rotation;
    }

    public float3 GetPredictedDirection()
    {
        TryUpdate();
        return SimulationBone.forward;
    }

    public bool FixPrediction;
    private int LastFrameUpdate = -1;
    private void TryUpdate()
    {
        if (LastFrameUpdate == Time.frameCount) { return; }
        LastFrameUpdate = Time.frameCount;

        // Trackers
        VRTrackersUpdate();

        // Direction
        DirectionMLPUpdate();

        // Simulation Bone ------
        // Position
        float3 hmdPosition = HMDBodyCenter.position;
        float3 groundTruthSimulationBonePosition = hmdPosition; // projection HMD
        groundTruthSimulationBonePosition.y = 0.0f;
        SimulationBone.position = groundTruthSimulationBonePosition;
        // Rotation
        float3 c0 = new float3(PredictedDirection[0], PredictedDirection[1], PredictedDirection[2]);
        float3 c1 = new float3(PredictedDirection[3], PredictedDirection[4], PredictedDirection[5]);
        quaternion simulationBoneRotation = MathExtensions.QuaternionFromContinuous(new float3x2(c0, c1)); // defined by simulation bone rotation (predicted or ground truth)
        simulationBoneRotation = math.mul(CurrentProjHMDRotation, simulationBoneRotation); // apply HMD rotation
        float3 scaledAxisRotation = MathExtensions.QuaternionToScaledAngleAxis(simulationBoneRotation);
        scaledAxisRotation.x = 0.0f;
        scaledAxisRotation.z = 0.0f;
        SimulationBone.rotation = MathExtensions.QuaternionFromScaledAngleAxis(scaledAxisRotation);
        if (FixPrediction)
        {
            if (PredictedDirection[0] > 1000 || PredictedDirection[1] > 1000 ||
                PredictedDirection[2] > 1000 || PredictedDirection[3] > 1000 ||
                PredictedDirection[4] > 1000 || PredictedDirection[5] > 1000)
            {
                float3x2 identity = MathExtensions.QuaternionToContinuous(quaternion.identity);
                PredictedDirection[0] = identity.c0[0];
                PredictedDirection[1] = identity.c0[1];
                PredictedDirection[2] = identity.c0[2];
                PredictedDirection[3] = identity.c1[0];
                PredictedDirection[4] = identity.c1[1];
                PredictedDirection[5] = identity.c1[2];
            }
        }

        // Debug
        if (ShowArrowIndicator && !ArrowDirectionIndicator.activeSelf)
        {
            ArrowDirectionIndicator.SetActive(true);
        }
        else if (!ShowArrowIndicator && ArrowDirectionIndicator.activeSelf)
        {
            ArrowDirectionIndicator.SetActive(false);
        }

        // DEBUG
        ArrowDirectionIndicator.transform.rotation = SimulationBone.rotation;
    }

    private void DirectionMLPUpdate()
    {
        // Normalize Input
        PoseDataset.NormalizeSimulationBoneRot(PredictedDirection);

        // Input
        for (int i = 0; i < CurrentTrackersInfo.Length; ++i)
        {
            DirectionMLPInput[i] = CurrentTrackersInfo[i];
        }
        int offset = CurrentTrackersInfo.Length;
        DirectionMLPInput[offset + 0] = PredictedDirection[0]; // Simulation Bone Rotation
        DirectionMLPInput[offset + 1] = PredictedDirection[1];
        DirectionMLPInput[offset + 2] = PredictedDirection[2];
        DirectionMLPInput[offset + 3] = PredictedDirection[3];
        DirectionMLPInput[offset + 4] = PredictedDirection[4];
        DirectionMLPInput[offset + 5] = PredictedDirection[5];
        // Predict
        DirectionMLP.Predict(DirectionMLPInput, PredictedDirection);

        // Denormalize
        PoseDataset.DenormalizeSimulationBoneRot(PredictedDirection); // Mean and Std should be from the training set
    }

    private void VRTrackersUpdate()
    {
        float3x2 hmdRot = new float3x2();
        float3 hmdVel = new float3();
        float3 hmdAngVel = new float3();
        if (HMDDevice != null)
        {
            quaternion hmdRotQuat = math.mul(math.mul(HMDDevice.rotation, math.inverse(TrackersDataset.LocalToVRSpace[0])), TrackersDataset.VRSpaceToTracker[0]);
            float3 hmdDir = math.mul(hmdRotQuat, math.forward());
            hmdDir.y = 0;
            hmdDir = math.normalize(hmdDir);
            CurrentProjHMDRotation = quaternion.LookRotation(hmdDir, math.up());
            hmdRotQuat = math.mul(math.inverse(CurrentProjHMDRotation), hmdRotQuat);
            hmdRot = MathExtensions.QuaternionToContinuous(hmdRotQuat);
            hmdVel = math.mul(math.inverse(CurrentProjHMDRotation), GetSmoothedVelocity(0, HMDDevice));
            hmdAngVel = math.mul(math.inverse(CurrentProjHMDRotation), GetSmoothedAngularVelocity(0, HMDDevice));
        }

        float3x2 leftHandRot = new float3x2();
        float3 leftHandVel = new float3();
        float3 leftHandAngVel = new float3();
        if (LeftController != null)
        {
            quaternion leftHandRotQuat = math.mul(math.mul(LeftController.rotation, math.inverse(TrackersDataset.LocalToVRSpace[1])), TrackersDataset.VRSpaceToTracker[1]);
            leftHandRotQuat = math.mul(math.inverse(CurrentProjHMDRotation), leftHandRotQuat);
            leftHandRot = MathExtensions.QuaternionToContinuous(leftHandRotQuat);
            leftHandVel = math.mul(math.inverse(CurrentProjHMDRotation), GetSmoothedVelocity(1, LeftController));
            leftHandAngVel = math.mul(math.inverse(CurrentProjHMDRotation), GetSmoothedAngularVelocity(1, LeftController));
        }

        float3x2 rightHandRot = new float3x2();
        float3 rightHandVel = new float3();
        float3 rightHandAngVel = new float3();
        if (RightController != null)
        {
            quaternion rightHandRotQuat = math.mul(math.mul(RightController.rotation, math.inverse(TrackersDataset.LocalToVRSpace[2])), TrackersDataset.VRSpaceToTracker[2]);
            rightHandRotQuat = math.mul(math.inverse(CurrentProjHMDRotation), rightHandRotQuat);
            rightHandRot = MathExtensions.QuaternionToContinuous(rightHandRotQuat);
            rightHandVel = math.mul(math.inverse(CurrentProjHMDRotation), GetSmoothedVelocity(2, RightController));
            rightHandAngVel = math.mul(math.inverse(CurrentProjHMDRotation), GetSmoothedAngularVelocity(2, RightController));
        }

        // Set Feature Vector
        AddTrackerCurrentFeatureVector(0, hmdRot, hmdVel, hmdAngVel);
        AddTrackerCurrentFeatureVector(1, leftHandRot, leftHandVel, leftHandAngVel);
        AddTrackerCurrentFeatureVector(2, rightHandRot, rightHandVel, rightHandAngVel);

        // Update Trackers Transforms (Debug)
        TrackerTransforms[0].position = HMDDevice.position;
        TrackerTransforms[0].rotation = math.mul(CurrentProjHMDRotation, MathExtensions.QuaternionFromContinuous(hmdRot));
        TrackerTransforms[1].position = LeftController.position;
        TrackerTransforms[1].rotation = math.mul(CurrentProjHMDRotation, MathExtensions.QuaternionFromContinuous(leftHandRot));
        TrackerTransforms[2].position = RightController.position;
        TrackerTransforms[2].rotation = math.mul(CurrentProjHMDRotation, MathExtensions.QuaternionFromContinuous(rightHandRot));

        // Normalize Feature Vector
        TrackersDataset.Normalize(CurrentTrackersInfo);

        // Update Previous
        PreviousPositions[0] = HMDDevice.position;
        PreviousRotations[0] = HMDDevice.rotation;
        PreviousPositions[1] = LeftController.position;
        PreviousRotations[1] = LeftController.rotation;
        PreviousPositions[2] = RightController.position;
        PreviousRotations[2] = RightController.rotation;
    }

    private void AddTrackerCurrentFeatureVector(int trackerIndex, float3x2 rot, float3 vel, float3 angVel)
    {
        int offset = trackerIndex * TrackersDataset.NumberFeaturesPerTracker;
        CurrentTrackersInfo[offset + 0] = rot.c0.x;
        CurrentTrackersInfo[offset + 1] = rot.c0.y;
        CurrentTrackersInfo[offset + 2] = rot.c0.z;
        CurrentTrackersInfo[offset + 3] = rot.c1.x;
        CurrentTrackersInfo[offset + 4] = rot.c1.y;
        CurrentTrackersInfo[offset + 5] = rot.c1.z;
        CurrentTrackersInfo[offset + 6] = vel.x;
        CurrentTrackersInfo[offset + 7] = vel.y;
        CurrentTrackersInfo[offset + 8] = vel.z;
        CurrentTrackersInfo[offset + 9] = angVel.x;
        CurrentTrackersInfo[offset + 10] = angVel.y;
        CurrentTrackersInfo[offset + 11] = angVel.z;
    }

    private float3 GetSmoothedVelocity(int trackerIndex, Transform device)
    {
        float dt = 1.0f / FixedFramerate;
        float3 currentPos = (float3)device.position;
        float3 currentVelocity = (currentPos - PreviousPositions[trackerIndex]) / dt; // pretend it's fixed frame rate

        // Move everything to the right (the leftmost is the newest)
        for (int i = PreviousVelocities[trackerIndex].Length - 1; i > 0; i--)
        {
            PreviousVelocities[trackerIndex][i] = PreviousVelocities[trackerIndex][i - 1];
        }
        // Add the current velocity
        PreviousVelocities[trackerIndex][0] = currentVelocity;

        float3 sum = float3.zero;
        for (int i = 0; i < PreviousVelocities[trackerIndex].Length; ++i)
        {
            sum += PreviousVelocities[trackerIndex][i];
        }
        currentVelocity = sum / NumberPastFramesVelocities;
        return currentVelocity;
    }

    private float3 GetSmoothedAngularVelocity(int trackerIndex, Transform device)
    {
        float dt = 1.0f / FixedFramerate;
        quaternion currentRot = (quaternion)device.rotation;
        float3 angularVelocity = MathExtensions.AngularVelocity(PreviousRotations[trackerIndex], currentRot, dt);

        // Move everything to the right (the leftmost is the newest)
        for (int i = PreviousAngularVelocities[trackerIndex].Length - 1; i > 0; i--)
        {
            PreviousAngularVelocities[trackerIndex][i] = PreviousAngularVelocities[trackerIndex][i - 1];
        }
        // Add the current angular velocity
        PreviousAngularVelocities[trackerIndex][0] = angularVelocity;

        float3 sum = float3.zero;
        for (int i = 0; i < PreviousAngularVelocities[trackerIndex].Length; ++i)
        {
            sum += PreviousAngularVelocities[trackerIndex][i];
        }
        angularVelocity = sum / NumberPastFramesVelocities;
        return angularVelocity;
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        // Skeleton
        if (SimulationBone == null) return;

        // Character
        Vector3 characterOrigin = SimulationBone.position; // Simulation Bone World Position
        Vector3 characterForward = SimulationBone.forward; // Simulation Bone World Rotation
        Gizmos.color = PredictColor;
        GizmosExtensions.DrawArrow(characterOrigin, characterOrigin + characterForward, thickness: 3);
    }
#endif
}
