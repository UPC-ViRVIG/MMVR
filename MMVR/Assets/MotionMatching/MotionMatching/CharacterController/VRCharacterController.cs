using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using MotionMatching;

using TrajectoryFeature = MotionMatching.MotionMatchingData.TrajectoryFeature;

public class VRCharacterController : MotionMatchingCharacterController
{
    [Header("Input Devices")]
    public Transform HMDDevice;

    // General ----------------------------------------------------------
    [Header("General")]
    public bool UseHMDForward = false;
    [Range(0.0f, 1.0f)] public float ResponsivenessPositions = 0.75f;
    [Range(0.0f, 1.0f)] public float ResponsivenessDirections = 0.75f;
    [Range(0.0f, 1.0f)] public float ThresholdNotifyVelocityChange = 0.1f;
    [Header("DEBUG")]
    public bool DebugCurrent = true;
    public bool DebugPrediction = true;
    public bool DebugClamping = true;
    // Adjustment & Clamping --------------------------------------------
    [Header("Adjustment")] // Move Simulation Bone towards the Simulation Object (motion matching towards character controller)
    public bool DoAdjustment = true;
    [Range(0.0f, 2.0f)] public float PositionAdjustmentHalflife = 0.1f; // Time needed to move half of the distance between SimulationBone and SimulationObject
    [Range(0.0f, 2.0f)] public float RotationAdjustmentHalflife = 0.1f;
    [Range(0.0f, 2.0f)] public float PosMaximumAdjustmentRatio = 0.1f; // Ratio between the adjustment and the character's velocity to clamp the adjustment
    [Range(0.0f, 2.0f)] public float RotMaximumAdjustmentRatio = 0.1f; // Ratio between the adjustment and the character's velocity to clamp the adjustment
    public bool DoClamping = true;
    [Range(0.0f, 2.0f)] public float MaxDistanceSimulationBoneAndObject = 0.1f; // Max distance between SimulationBone and SimulationObject
    // --------------------------------------------------------------------------

    private Tracker HMDTracker;
    private float3 PositionHMD; // Position of the Simulation Object (controller) for HMD
    private quaternion RotationHMD; // Rotation of the Simulation Object (controller) for HMD
    private int HeadJointIndex;
    private float PreviousHMDDesiredSpeedSq;
    private VRDirectionPredictor DirectionPredictor;


    // FUNCTIONS ---------------------------------------------------------------
    private void Awake()
    {
        DirectionPredictor = GetComponent<VRDirectionPredictor>();
    }

    private void Start()
    {
        HMDTracker = new Tracker(HMDDevice, this);

        PositionHMD = new float3();
        RotationHMD = new quaternion();

        if (!SimulationBone.GetSkeleton().Find(HumanBodyBones.Head, out Skeleton.Joint headJoint))
        {
            Debug.LogError("Could not find head joint");
        }
        HeadJointIndex = headJoint.Index;

        Application.targetFrameRate = AverageFPS;
    }

    protected override void OnUpdate()
    {
        Tracker tracker = HMDTracker;
        float3 currentPos = GetCurrentHMDPosition();
        quaternion currentRot = GetCurrentHMDRotation();

        // Input
        float3 desiredVelocity = tracker.GetSmoothedVelocity();
        float sqDesiredVelocity = math.lengthsq(desiredVelocity);
        if (sqDesiredVelocity - PreviousHMDDesiredSpeedSq > ThresholdNotifyVelocityChange * ThresholdNotifyVelocityChange)
        {
            NotifyInputChangedQuickly();
        }
        PreviousHMDDesiredSpeedSq = sqDesiredVelocity;
        tracker.DesiredRotation = UseHMDForward ? HMDDevice.rotation : DirectionPredictor.GetPredictedRotation();
        quaternion desiredRotation = tracker.DesiredRotation;

        // Rotations
        tracker.PredictRotations(currentRot, desiredRotation, AveragedDeltaTime);

        // Positions
        tracker.PredictPositions(currentPos, desiredVelocity, AveragedDeltaTime);

        // Update Character Controller
        PositionHMD = HMDTracker.Device.position;
        RotationHMD = tracker.ComputeNewRot(currentRot, desiredRotation);

        // Adjust SimulationBone to pull the character (moving SimulationBone) towards the Simulation Object (character controller)
        if (DoAdjustment) AdjustSimulationBone();
        if (DoClamping) ClampSimulationBone();

        DirectionPredictor.SetEnabledDebug(!UseHMDForward);
        DirectionPredictor.SetPositionDebug(SimulationBone.GetSkeletonTransforms()[0].position);
    }

    private void AdjustSimulationBone()
    {
        AdjustCharacterPosition();
        AdjustCharacterRotation();
    }

    private void ClampSimulationBone()
    {
        // Clamp Position
        float3 simulationObject = HMDTracker.Device.position;
        simulationObject.y = 0.0f;
        float3 simulationBone = SimulationBone.GetSkeletonTransforms()[0].position;
        simulationBone.y = 0.0f;
        if (math.distance(simulationObject, simulationBone) > MaxDistanceSimulationBoneAndObject)
        {
            float3 newSimulationBonePos = MaxDistanceSimulationBoneAndObject * math.normalize(simulationBone - simulationObject) + simulationObject;
            SimulationBone.SetPosAdjustment(newSimulationBonePos - simulationBone);
        }
    }

    private void AdjustCharacterPosition()
    {
        float3 simulationObject = HMDTracker.Device.position;
        float3 simulationBone = SimulationBone.GetSkeletonTransforms()[0].position;
        float3 differencePosition = simulationObject - simulationBone;
        differencePosition.y = 0; // No vertical Axis
        // Damp the difference using the adjustment halflife and dt
        float3 adjustmentPosition = Spring.DampAdjustmentImplicit(differencePosition, PositionAdjustmentHalflife, Time.deltaTime);
        // Clamp adjustment if the length is greater than the character velocity
        // multiplied by the ratio
        float maxLength = PosMaximumAdjustmentRatio * math.length(SimulationBone.Velocity) * Time.deltaTime;
        if (math.length(adjustmentPosition) > maxLength)
        {
            adjustmentPosition = maxLength * math.normalize(adjustmentPosition);
        }
        // Move the simulation bone towards the simulation object
        SimulationBone.SetPosAdjustment(adjustmentPosition);
    }

    private void AdjustCharacterRotation()
    {
        float3 simulationObject = HMDTracker.Device.TransformDirection(SimulationBone.MMData.GetLocalForward(0));
        float3 simulationBone = SimulationBone.GetSkeletonTransforms()[0].forward;
        // Only Y Axis rotation
        simulationObject.y = 0;
        simulationObject = math.normalize(simulationObject);
        simulationBone.y = 0;
        simulationBone = math.normalize(simulationBone);
        // Find the difference in rotation (from character to simulation object)
        quaternion differenceRotation = MathExtensions.FromToRotation(simulationBone, simulationObject, math.up());
        // Damp the difference using the adjustment halflife and dt
        quaternion adjustmentRotation = Spring.DampAdjustmentImplicit(differenceRotation, RotationAdjustmentHalflife, Time.deltaTime);
        // Clamp adjustment if the length is greater than the character angular velocity
        // multiplied by the ratio
        float maxLength = RotMaximumAdjustmentRatio * math.length(SimulationBone.AngularVelocity) * Time.deltaTime;
        if (math.length(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)) > maxLength)
        {
            adjustmentRotation = MathExtensions.QuaternionFromScaledAngleAxis(maxLength * math.normalize(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)));
        }
        // Rotate the simulation bone towards the simulation object
        SimulationBone.SetRotAdjustment(adjustmentRotation);
    }

    private float3 GetCurrentHMDPosition()
    {
        return PositionHMD;
    }
    private quaternion GetCurrentHMDRotation()
    {
        return RotationHMD;
    }

    private float3 GetWorldSpacePosition(int predictionIndex)
    {
        Tracker tracker = HMDTracker;
        return tracker.PredictedPosition[predictionIndex];
    }

    private float3 GetWorldSpaceDirectionPrediction(int index)
    {
        Tracker tracker = HMDTracker;
        float3 dir = math.mul(tracker.PredictedRotations[index], math.forward());
        return math.normalize(dir);
    }

    public override float3 GetWorldInitPosition()
    {
        return transform.position;
    }
    public override float3 GetWorldInitDirection()
    {
        return transform.forward;
    }

    public override float3 GetWorldSpacePrediction(MotionMatchingData.TrajectoryFeature feature, int predictionIndex)
    {
        Debug.Assert(feature.Project == true, "Project must be true");
        switch (feature.Bone)
        {
            case HumanBodyBones.Head:
                break;
            case HumanBodyBones.LeftHand:
                Debug.Assert(false, "LeftHand is not supported");
                break;
            case HumanBodyBones.RightHand:
                Debug.Assert(false, "RightHand is not supported");
                break;
            default:
                Debug.Assert(false, "Unknown Bone: " + feature.Bone);
                break;
        }
        switch (feature.FeatureType)
        {
            case TrajectoryFeature.Type.Position:
                float3 pos = GetWorldSpacePosition(predictionIndex);
                pos.y = 0.0f; // Project to the ground
                return pos;
            case TrajectoryFeature.Type.Direction:
                float3 dir = GetWorldSpaceDirectionPrediction(predictionIndex);
                dir.y = 0.0f; // Project to the ground
                dir = math.normalize(dir);
                return dir;
            default:
                Debug.Assert(false, "Unknown feature type: " + feature.FeatureType);
                break;
        }
        return float3.zero;
    }

    private class Tracker
    {
        public Transform Device;
        public VRCharacterController Controller;
        // Rotation and Predicted Rotation ------------------------------------------
        public quaternion DesiredRotation;
        public quaternion[] PredictedRotations;
        public float3 AngularVelocity;
        public float3[] PredictedAngularVelocities;
        // Position and Predicted Position ------------------------------------------
        public float3[] PredictedPosition;
        public float3 Velocity;
        public float3 Acceleration;
        public float3[] PredictedVelocity;
        public float3[] PredictedAcceleration;
        // Features -----------------------------------------------------------------
        public int[] TrajectoryPosPredictionFrames;
        public int[] TrajectoryRotPredictionFrames;
        public int NumberPredictionPos { get { return TrajectoryPosPredictionFrames.Length; } }
        public int NumberPredictionRot { get { return TrajectoryRotPredictionFrames.Length; } }
        // Previous -----------------------------------------------------------------
        public float3 PrevInputPos;
        public quaternion PrevInputRot;
        public float3[] PreviousVelocities;
        public int PreviousVelocitiesIndex;
        public float3[] PreviousAngularVelocities;
        public int PreviousAngularVelocitiesIndex;
        public int NumberPastFrames = 1;
        // --------------------------------------------------------------------------

        public Tracker(Transform device, VRCharacterController controller)
        {
            Device = device;
            Controller = controller;
            PrevInputPos = (float3)Device.position;
            PreviousVelocities = new float3[NumberPastFrames]; // HARDCODED
            PreviousAngularVelocities = new float3[NumberPastFrames]; // HARDCODED

            TrajectoryPosPredictionFrames = new int[] { 20, 40, 60 }; // HARDCODED
            TrajectoryRotPredictionFrames = new int[] { 20, 40, 60 }; // HARDCODED
                                                                      // TODO: generalize this... allow different number of prediction frames for different features
            Debug.Assert(TrajectoryPosPredictionFrames.Length == TrajectoryRotPredictionFrames.Length, "Trajectory Position and Trajectory Direction Prediction Frames must be the same for SpringCharacterController");
            for (int i = 0; i < TrajectoryPosPredictionFrames.Length; ++i)
            {
                Debug.Assert(TrajectoryPosPredictionFrames[i] == TrajectoryRotPredictionFrames[i], "Trajectory Position and Trajectory Direction Prediction Frames must be the same for SpringCharacterController");
            }
            //if (Controller.AverageFPS != TrajectoryPosPredictionFrames[TrajectoryPosPredictionFrames.Length - 1]) Debug.LogWarning("AverageFPS is not the same as the last Prediction Frame... maybe you forgot changing the hardcoded value?");
            //if (Controller.AverageFPS != TrajectoryRotPredictionFrames[TrajectoryRotPredictionFrames.Length - 1]) Debug.LogWarning("AverageFPS is not the same as the last Prediction Frame... maybe you forgot changing the hardcoded value?");


            PredictedPosition = new float3[NumberPredictionPos];
            PredictedVelocity = new float3[NumberPredictionPos];
            PredictedAcceleration = new float3[NumberPredictionPos];
            PredictedRotations = new quaternion[NumberPredictionRot];
            PredictedAngularVelocities = new float3[NumberPredictionRot];
        }

        public void PredictRotations(quaternion currentRotation, quaternion desiredRotation, float averagedDeltaTime)
        {
            for (int i = 0; i < NumberPredictionRot; i++)
            {
                // Init Predicted values
                PredictedRotations[i] = currentRotation;
                PredictedAngularVelocities[i] = AngularVelocity;
                // Predict
                Spring.SimpleSpringDamperImplicit(ref PredictedRotations[i], ref PredictedAngularVelocities[i],
                                                  desiredRotation, 1.0f - Controller.ResponsivenessDirections, TrajectoryRotPredictionFrames[i] * averagedDeltaTime);
            }
        }

        /* https://theorangeduck.com/page/spring-roll-call#controllers */
        public void PredictPositions(float3 currentPos, float3 desiredVelocity, float averagedDeltaTime)
        {
            int lastPredictionFrames = 0;
            for (int i = 0; i < NumberPredictionPos; ++i)
            {
                if (i == 0)
                {
                    PredictedPosition[i] = currentPos;
                    PredictedVelocity[i] = Velocity;
                    PredictedAcceleration[i] = Acceleration;
                }
                else
                {
                    PredictedPosition[i] = PredictedPosition[i - 1];
                    PredictedVelocity[i] = PredictedVelocity[i - 1];
                    PredictedAcceleration[i] = PredictedAcceleration[i - 1];
                }
                int diffPredictionFrames = TrajectoryPosPredictionFrames[i] - lastPredictionFrames;
                lastPredictionFrames = TrajectoryPosPredictionFrames[i];
                Spring.CharacterPositionUpdate(ref PredictedPosition[i], ref PredictedVelocity[i], ref PredictedAcceleration[i],
                                               desiredVelocity, 1.0f - Controller.ResponsivenessPositions, diffPredictionFrames * averagedDeltaTime);
            }
        }

        public quaternion ComputeNewRot(quaternion currentRotation, quaternion desiredRotation)
        {
            quaternion newRotation = currentRotation;
            Spring.SimpleSpringDamperImplicit(ref newRotation, ref AngularVelocity, desiredRotation, 1.0f - Controller.ResponsivenessDirections, Time.deltaTime);
            return newRotation;
        }

        public float3 GetSmoothedVelocity()
        {
            float dt = Controller.AveragedDeltaTime;
            float3 currentInputPos = (float3)Device.position;
            float3 currentSpeed = (currentInputPos - PrevInputPos) / dt; // pretend it's fixed frame rate
            PrevInputPos = currentInputPos;

            PreviousVelocities[PreviousVelocitiesIndex] = currentSpeed;
            PreviousVelocitiesIndex = (PreviousVelocitiesIndex + 1) % PreviousVelocities.Length;

            float3 sum = float3.zero;
            for (int i = 0; i < PreviousVelocities.Length; ++i)
            {
                sum += PreviousVelocities[i];
            }
            currentSpeed = sum / NumberPastFrames;
            return currentSpeed;
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        const float radius = 0.05f;
        const float vectorReduction = 0.5f;

        Tracker tracker = HMDTracker;

        Vector3 transformPos = (Vector3)GetCurrentHMDPosition();
        transformPos.y = 0.0f;
        if (DebugCurrent)
        {
            // Draw Current Position & Velocity
            Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
            Gizmos.DrawSphere(transformPos, radius);
            GizmosExtensions.DrawLine(transformPos, transformPos + ((Quaternion)GetCurrentHMDRotation() * Vector3.forward) * vectorReduction, 3);
        }

        if (DebugPrediction)
        {
            // Draw Predicted Position & Velocity
            Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
            for (int i = 0; i < tracker.PredictedPosition.Length; ++i)
            {
                float3 predictedPos = tracker.PredictedPosition[i];
                predictedPos.y = 0.0f;
                float3 predictedDir3D = GetWorldSpaceDirectionPrediction(i);
                predictedDir3D.y = 0.0f;
                predictedDir3D = math.normalize(predictedDir3D);
                Gizmos.DrawSphere(predictedPos, radius);
                GizmosExtensions.DrawLine(predictedPos, predictedPos + predictedDir3D * vectorReduction, 3);
            }
        }

        if (DebugClamping)
        {
            // Draw Clamp Circle
            if (DoClamping)
            {
                Gizmos.color = new Color(0.1f, 1.0f, 0.1f, 1.0f);
                GizmosExtensions.DrawWireCircle(transformPos, MaxDistanceSimulationBoneAndObject, quaternion.identity);
            }
        }
    }
#endif
}

