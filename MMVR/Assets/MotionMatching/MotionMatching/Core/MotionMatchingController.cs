using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;

namespace MotionMatching
{
    using TrajectoryFeature = MotionMatchingData.TrajectoryFeature;
    using PoseFeature = MotionMatchingData.PoseFeature;

    // Simulation bone is the transform
    public class MotionMatchingController : MonoBehaviour
    {
        public event Action OnSkeletonTransformUpdated;

        public MotionMatchingCharacterController CharacterController;
        public MotionMatchingData MMData;
        public Calibrator Calibrator;
        public SquatDataset[] SquatDatasets;
        public bool LockFPS = true;
        public int SearchFrames = 10; // Motion Matching every SearchFrames frames
        public bool Inertialize = true; // Should inertialize transitions after a big change of the pose
        public bool FootLock = true; // Should lock the feet to the ground when contact information is true
        public float FootUnlockDistance = 0.2f; // Distance from actual pose to IK target to unlock the feet
        [Range(0.0f, 1.0f)] public float InertializeHalfLife = 0.1f; // Time needed to move half of the distance between the source to the target pose
        [Tooltip("How important is the trajectory (future positions + future directions)")][Range(0.0f, 1.0f)] public float Responsiveness = 1.0f;
        [Tooltip("How important is the current pose")][Range(0.0f, 1.0f)] public float Quality = 1.0f;
        [HideInInspector] public float[] FeatureWeights;
        [Header("Debug")]
        public float SpheresRadius = 0.1f;
        public bool DebugSkeleton = true;
        public bool DebugCurrent = true;
        public bool DebugPose = true;
        public bool DebugTrajectory = true;
        public bool DebugContacts = true;
        public GameObject CurrentRotationPrefab;
        public bool DebugCurrentRotation = true;


        public float3 Velocity { get; private set; }
        public float3 AngularVelocity { get; private set; }
        public float FrameTime { get; private set; }

        private PoseSet PoseSet;
        private FeatureSet FeatureSet;
        private PoseSet[] SquatPoseSets;
        private FeatureSet[] SquatFeatureSets;


        private Transform[] SkeletonTransforms;
        private float3 AnimationSpaceOriginPos;
        private quaternion InverseAnimationSpaceOriginRot;
        private float3 MMTransformOriginPose; // Position of the transform right after motion matching search
        private quaternion MMTransformOriginRot; // Rotation of the transform right after motion matching search
        private int LastMMSearchFrame; // Frame before the last Motion Matching Search
        public int CurrentFrame { get; private set; } // Current frame index in the pose/feature set
        private int SearchFrameCount;
        private NativeArray<float> QueryFeature;
        private NativeArray<int> SearchResult;
        private NativeArray<float> FeaturesWeightsNativeArray;
        private Inertialization Inertialization;
        // Foot Lock
        private bool IsLeftFootContact, IsRightFootContact;
        private float3 LeftToesContactTarget, RightToesContactTarget; // Target position of the toes
        private float3 LeftFootContact, RightFootContact; // Position of the foot
        private float3 LeftFootPoleContact, RightFootPoleContact; // Forward vector of the knee
        private float3 LeftLowerLegLocalForward, RightLowerLegLocalForward;
        private int LeftToesIndex, LeftFootIndex, LeftLowerLegIndex, LeftUpperLegIndex;
        private int RightToesIndex, RightFootIndex, RightLowerLegIndex, RightUpperLegIndex;
        // Squat
        private int SquatIndex; // 0 - No Squat, 1 - First Squat Dataset, 2 - Second...
        // Debug
        private GameObject CurrentRotationIndicator;


        private void Awake()
        {
            // PoseSet
            PoseSet = MMData.GetOrImportPoseSet();
            SquatPoseSets = new PoseSet[SquatDatasets.Length];
            for (int i = 0; i < SquatDatasets.Length; i++)
            {
                SquatPoseSets[i] = SquatDatasets[i].MMData.GetOrImportPoseSet();
            }

            // FeatureSet
            FeatureSet = MMData.GetOrImportFeatureSet();
            SquatFeatureSets = new FeatureSet[SquatDatasets.Length];
            for (int i = 0; i < SquatDatasets.Length; i++)
            {
                SquatFeatureSets[i] = SquatDatasets[i].MMData.GetOrImportFeatureSet();
            }

            // Skeleton
            SkeletonTransforms = new Transform[PoseSet.Skeleton.Joints.Count];
            SkeletonTransforms[0] = transform; // Simulation Bone
            for (int j = 1; j < PoseSet.Skeleton.Joints.Count; j++)
            {
                // Joints
                Skeleton.Joint joint = PoseSet.Skeleton.Joints[j];
                Transform t = (new GameObject()).transform;
                t.name = joint.Name;
                t.SetParent(SkeletonTransforms[joint.ParentIndex], false);
                t.localPosition = joint.LocalOffset;
                SkeletonTransforms[j] = t;
            }

            // Inertialization
            Inertialization = new Inertialization(PoseSet.Skeleton);

            // FPS
            FrameTime = PoseSet.FrameTime;
            if (LockFPS)
            {
                Application.targetFrameRate = Mathf.RoundToInt(1.0f / FrameTime);
                Debug.Log("[Motion Matching] Updated Target FPS: " + Application.targetFrameRate);
            }
            else
            {
                Application.targetFrameRate = -1;
            }

            // Other initialization
            SearchResult = new NativeArray<int>(1, Allocator.Persistent);
            int numberFeatures = (MMData.TrajectoryFeatures.Count + MMData.PoseFeatures.Count);
            if (FeatureWeights == null || FeatureWeights.Length != numberFeatures)
            {
                float[] newWeights = new float[numberFeatures];
                for (int i = 0; i < newWeights.Length; ++i) newWeights[i] = 1.0f;
                for (int i = 0; i < Mathf.Min(FeatureWeights.Length, newWeights.Length); i++) newWeights[i] = FeatureWeights[i];
                FeatureWeights = newWeights;
            }
            FeaturesWeightsNativeArray = new NativeArray<float>(FeatureSet.FeatureSize, Allocator.Persistent);
            QueryFeature = new NativeArray<float>(FeatureSet.FeatureSize, Allocator.Persistent);
            // Search first Frame valid (to start with a valid pose)
            for (int i = 0; i < FeatureSet.NumberFeatureVectors; i++)
            {
                if (FeatureSet.IsValidFeature(i))
                {
                    LastMMSearchFrame = i;
                    CurrentFrame = i;
                    break;
                }
            }
            // Foot Lock
            if (!PoseSet.Skeleton.Find(HumanBodyBones.LeftToes, out Skeleton.Joint leftToesJoint)) Debug.LogError("[Motion Matching] LeftToes not found");
            LeftToesIndex = leftToesJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.LeftFoot, out Skeleton.Joint leftFootJoint)) Debug.LogError("[Motion Matching] LeftFoot not found");
            LeftFootIndex = leftFootJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.LeftLowerLeg, out Skeleton.Joint leftLowerLegJoint)) Debug.LogError("[Motion Matching] LeftLowerLeg not found");
            LeftLowerLegIndex = leftLowerLegJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.LeftUpperLeg, out Skeleton.Joint leftUpperLegJoint)) Debug.LogError("[Motion Matching] LeftUpperLeg not found");
            LeftUpperLegIndex = leftUpperLegJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.RightToes, out Skeleton.Joint rightToesJoint)) Debug.LogError("[Motion Matching] RightToes not found");
            RightToesIndex = rightToesJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.RightFoot, out Skeleton.Joint rightFootJoint)) Debug.LogError("[Motion Matching] RightFoot not found");
            RightFootIndex = rightFootJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.RightLowerLeg, out Skeleton.Joint rightLowerLegJoint)) Debug.LogError("[Motion Matching] RightLowerLeg not found");
            RightLowerLegIndex = rightLowerLegJoint.Index;
            if (!PoseSet.Skeleton.Find(HumanBodyBones.RightUpperLeg, out Skeleton.Joint rightUpperLegJoint)) Debug.LogError("[Motion Matching] RightUpperLeg not found");
            RightUpperLegIndex = rightUpperLegJoint.Index;
            LeftLowerLegLocalForward = MMData.GetLocalForward(LeftLowerLegIndex);
            RightLowerLegLocalForward = MMData.GetLocalForward(RightLowerLegIndex);
            // Init Pose
            SkeletonTransforms[0].position = CharacterController.GetWorldInitPosition();
            SkeletonTransforms[0].rotation = quaternion.LookRotation(CharacterController.GetWorldInitDirection(), Vector3.up);
            // Debug
            CurrentRotationIndicator = Instantiate(CurrentRotationPrefab, SkeletonTransforms[0]);
            CurrentRotationIndicator.transform.localPosition = Vector3.zero;
        }

        private void OnEnable()
        {
            SearchFrameCount = 0;
            CharacterController.OnUpdated += OnCharacterControllerUpdated;
            CharacterController.OnInputChangedQuickly += OnInputChangedQuickly;
        }

        private void OnDisable()
        {
            CharacterController.OnUpdated -= OnCharacterControllerUpdated;
            CharacterController.OnInputChangedQuickly -= OnInputChangedQuickly;
        }

        private void OnCharacterControllerUpdated(float deltaTime)
        {
            PROFILE.BEGIN_SAMPLE_PROFILING("Motion Matching Total");
            if (SearchFrameCount == 0)
            {
                // Determine squat level
                int prevSquatIndex = SquatIndex;
                float heightPercentage = Calibrator.HMDHeight / Calibrator.EyesHeight;
                int newSquatIndex = 0;
                for (int i = 0; i < SquatDatasets.Length; i++)
                {
                    float p = SquatDatasets[i].PercentageHeight;
                    if ((p < 1.0f && heightPercentage < p) || (p > 1.0f && heightPercentage > p))
                    {
                        newSquatIndex = i + 1; // 0 is the default
                    }
                }
                SquatIndex = newSquatIndex;

                // Motion Matching
                PROFILE.BEGIN_SAMPLE_PROFILING("Motion Matching Search");
                int bestFrame = SearchMotionMatching(prevSquatIndex);
                PROFILE.END_SAMPLE_PROFILING("Motion Matching Search");
                if (bestFrame != CurrentFrame || prevSquatIndex != SquatIndex)
                {
                    PoseSet sourcePoseSet = PoseSet;
                    if (prevSquatIndex > 0) sourcePoseSet = SquatPoseSets[prevSquatIndex - 1];
                    PoseSet targetPoseSet = PoseSet;
                    if (SquatIndex > 0) targetPoseSet = SquatPoseSets[SquatIndex - 1];
                    // Inertialize
                    if (Inertialize)
                    {
                        Inertialization.PoseTransition(sourcePoseSet, targetPoseSet, CurrentFrame, bestFrame);
                    }
                    LastMMSearchFrame = CurrentFrame;
                    CurrentFrame = bestFrame;
                    // Update Current Animation Space Origin
                    targetPoseSet.GetPose(CurrentFrame, out PoseVector mmPose);
                    AnimationSpaceOriginPos = mmPose.JointLocalPositions[0];
                    InverseAnimationSpaceOriginRot = math.inverse(mmPose.JointLocalRotations[0]);
                    MMTransformOriginPose = SkeletonTransforms[0].position;
                    MMTransformOriginRot = SkeletonTransforms[0].rotation;
                }
                SearchFrameCount = SearchFrames;
            }
            else
            {
                // Advance
                SearchFrameCount -= 1;
            }
            // Always advance one (bestFrame from motion matching is the best match to the current frame, but we want to move to the next frame)
            CurrentFrame += 1;

            UpdateTransformAndSkeleton(CurrentFrame);
            PROFILE.END_SAMPLE_PROFILING("Motion Matching Total");
        }

        private void OnInputChangedQuickly()
        {
            SearchFrameCount = 0; // Force search
        }

        private int SearchMotionMatching(int previousSquatIndex)
        {
            // Weights
            UpdateAndGetFeatureWeights();

            // Previous FeatureSet
            FeatureSet previousFeatureSet = FeatureSet;
            if (previousSquatIndex > 0) previousFeatureSet = SquatFeatureSets[previousSquatIndex - 1];
            // Current FeatureSet
            FeatureSet currentFeatureSet = FeatureSet;
            if (SquatIndex > 0) currentFeatureSet = SquatFeatureSets[SquatIndex - 1];

            // Init Query Vector (with previous information)
            previousFeatureSet.GetFeature(QueryFeature, CurrentFrame);
            previousFeatureSet.DenormalizeFeatureVector(QueryFeature);
            // Now Current Feature Set
            currentFeatureSet.NormalizeFeatureVector(QueryFeature);
            FillTrajectory(QueryFeature, currentFeatureSet);

            // Get next feature vector (when doing motion matching search, they need less error than this)
            float currentDistance = float.MaxValue;
            bool currentValid = false;
            if (previousFeatureSet.IsValidFeature(CurrentFrame) && SquatIndex == previousSquatIndex)
            {
                currentValid = true;
                currentDistance = 0.0f;
                // the pose is the same... the distance is only the trajectory
                for (int j = 0; j < previousFeatureSet.PoseOffset; j++)
                {
                    float diff = previousFeatureSet.GetFeatures()[CurrentFrame * previousFeatureSet.FeatureSize + j] - QueryFeature[j];
                    currentDistance += diff * diff * FeaturesWeightsNativeArray[j];
                }
            }

            // Search
            var job = new LinearMotionMatchingSearchBurst
            {
                Valid = currentFeatureSet.GetValid(),
                Features = currentFeatureSet.GetFeatures(),
                QueryFeature = QueryFeature,
                FeatureWeights = FeaturesWeightsNativeArray,
                FeatureSize = currentFeatureSet.FeatureSize,
                PoseOffset = currentFeatureSet.PoseOffset,
                CurrentDistance = currentDistance,
                BestIndex = SearchResult
            };
            job.Schedule().Complete();

            // Check if use current or best
            int best = SearchResult[0];
            if (currentValid && best == -1) best = CurrentFrame;

            return best;
        }

        private void FillTrajectory(NativeArray<float> vector, FeatureSet featureSet)
        {
            int offset = 0;
            for (int i = 0; i < MMData.TrajectoryFeatures.Count; i++)
            {
                TrajectoryFeature feature = MMData.TrajectoryFeatures[i];
                for (int p = 0; p < feature.FramesPrediction.Length; ++p)
                {
                    float3 value = CharacterController.GetWorldSpacePrediction(feature, p);
                    switch (feature.FeatureType)
                    {
                        case TrajectoryFeature.Type.Position:
                            value = GetPositionLocalCharacter(value);
                            break;
                        case TrajectoryFeature.Type.Direction:
                            value = GetDirectionLocalCharacter(value);
                            break;
                        default:
                            Debug.Assert(false, "Unknown feature type: " + feature.FeatureType);
                            break;
                    }
                    if (feature.Project)
                    {
                        vector[offset + 0] = value.x;
                        vector[offset + 1] = value.z;
                        offset += 2;
                    }
                    else
                    {
                        vector[offset + 0] = value.x;
                        vector[offset + 1] = value.y;
                        vector[offset + 2] = value.z;
                        offset += 3;
                    }
                }
            }
            // Normalize (only trajectory... because current FeatureVector is already normalized)
            featureSet.NormalizeTrajectory(vector);
        }

        private void UpdateTransformAndSkeleton(int frameIndex)
        {
            PoseSet poseSet = PoseSet;
            if (SquatIndex > 0) poseSet = SquatPoseSets[SquatIndex - 1];
            poseSet.GetPose(frameIndex, out PoseVector pose);
            // Update Inertialize if enabled
            if (Inertialize)
            {
                Inertialization.Update(pose, InertializeHalfLife, Time.deltaTime);
            }
            // Simulation Bone
            float3 previousPosition = SkeletonTransforms[0].position;
            quaternion previousRotation = SkeletonTransforms[0].rotation;
            // animation space to local space
            float3 localSpacePos = math.mul(InverseAnimationSpaceOriginRot, pose.JointLocalPositions[0] - AnimationSpaceOriginPos);
            quaternion localSpaceRot = math.mul(InverseAnimationSpaceOriginRot, pose.JointLocalRotations[0]);
            // local space to world space
            SkeletonTransforms[0].position = math.mul(MMTransformOriginRot, localSpacePos) + MMTransformOriginPose;
            SkeletonTransforms[0].rotation = math.mul(MMTransformOriginRot, localSpaceRot);
            // update velocity and angular velocity
            Velocity = ((float3)SkeletonTransforms[0].position - previousPosition) / Time.deltaTime;
            AngularVelocity = MathExtensions.AngularVelocity(previousRotation, SkeletonTransforms[0].rotation, Time.deltaTime);
            // Joints
            if (Inertialize)
            {
                for (int i = 1; i < Inertialization.InertializedRotations.Length; i++)
                {
                    SkeletonTransforms[i].localRotation = Inertialization.InertializedRotations[i];
                }
            }
            else
            {
                for (int i = 1; i < pose.JointLocalRotations.Length; i++)
                {
                    SkeletonTransforms[i].localRotation = pose.JointLocalRotations[i];
                }
            }
            // Hips Position
            SkeletonTransforms[1].localPosition = Inertialize ? Inertialization.InertializedHips : pose.JointLocalPositions[1];
            // Foot Lock
            UpdateFootLock(pose);
            // Post processing the transforms
            if (OnSkeletonTransformUpdated != null) OnSkeletonTransformUpdated.Invoke();

            // Debug
            if (DebugCurrentRotation && !CurrentRotationIndicator.activeSelf)
            {
                CurrentRotationIndicator.SetActive(true);
            }
            if (!DebugCurrentRotation && CurrentRotationIndicator.activeSelf)
            {
                CurrentRotationIndicator.SetActive(false);
            }
        }

        private void UpdateFootLock(PoseVector pose)
        {
            float3 currentLeftToesPosition = SkeletonTransforms[LeftToesIndex].position;
            float3 currentRightToesPosition = SkeletonTransforms[RightToesIndex].position;
            // Compute input contact position velocity
            float3 currentLeftToesVelocity = (currentLeftToesPosition - (float3)LeftToesContactTarget) / Time.deltaTime;
            float3 currentRightToesVelocity = (currentRightToesPosition - (float3)RightToesContactTarget) / Time.deltaTime;
            LeftToesContactTarget = currentLeftToesPosition;
            RightToesContactTarget = currentRightToesPosition;

            // Update Inertializer
            Inertialization.UpdateContact(IsLeftFootContact ? LeftFootContact : currentLeftToesPosition,
                                          IsLeftFootContact ? float3.zero : currentLeftToesVelocity,
                                          IsRightFootContact ? RightFootContact : currentRightToesPosition,
                                          IsRightFootContact ? float3.zero : currentRightToesVelocity,
                                          InertializeHalfLife, Time.deltaTime);
            float3 leftContactPosition = Inertialization.InertializedLeftContact;
            float3 leftContactVelocity = Inertialization.InertializedLeftContactVelocity;
            float3 rightContactPosition = Inertialization.InertializedRightContact;
            float3 rightContactVelocity = Inertialization.InertializedRightContactVelocity;

            // If the contact point is too far from the current input position
            // unlock the contact
            bool unlockLeftContact = IsLeftFootContact && (math.length(LeftFootContact - currentLeftToesPosition) > FootUnlockDistance);
            bool unlockRightContact = IsRightFootContact && (math.length(RightFootContact - currentRightToesPosition) > FootUnlockDistance);

            // If the contact was previously inactive and now it is active,
            // transition to the locked contact state
            // Also, make sure the inertialization returns an almost 0 velocity before locking
            if (!IsLeftFootContact && pose.LeftFootContact && math.length(leftContactVelocity) < MMData.ContactVelocityThreshold)
            {
                // Contact point is the current position of the foot
                // projected onto the ground + foot height
                IsLeftFootContact = true;
                LeftFootContact = leftContactPosition;
                // LeftFootContact.y = 0.0f;
                Transform leftLowerLeg = SkeletonTransforms[LeftLowerLegIndex];
                LeftFootPoleContact = math.mul(leftLowerLeg.rotation, LeftLowerLegLocalForward);

                if (Inertialize)
                {
                    Inertialization.LeftContactTransition(currentLeftToesPosition, currentLeftToesVelocity, LeftFootContact, float3.zero);
                }
                else
                {
                    Inertialization.LeftContactTransition(currentLeftToesPosition, currentLeftToesVelocity, currentLeftToesPosition, currentLeftToesVelocity);
                }
            }
            // If we need to unlock or previously in contact but now not
            // we transition to the input position
            else if (unlockLeftContact || (IsLeftFootContact && !pose.LeftFootContact))
            {
                IsLeftFootContact = false;

                if (Inertialize)
                {
                    Inertialization.LeftContactTransition(LeftFootContact, float3.zero, currentLeftToesPosition, currentLeftToesVelocity);
                }
                else
                {
                    Inertialization.LeftContactTransition(currentLeftToesPosition, currentLeftToesVelocity, currentLeftToesPosition, currentLeftToesVelocity);
                }
            }

            // Same for Right Foot
            if (!IsRightFootContact && pose.RightFootContact && math.length(rightContactVelocity) < MMData.ContactVelocityThreshold)
            {
                IsRightFootContact = true;
                RightFootContact = rightContactPosition;
                // RightFootContact.y = 0.0f;
                Transform rightLowerLeg = SkeletonTransforms[RightLowerLegIndex];
                RightFootPoleContact = math.mul(rightLowerLeg.rotation, RightLowerLegLocalForward);

                if (Inertialize)
                {
                    Inertialization.RightContactTransition(currentRightToesPosition, currentRightToesVelocity, RightFootContact, float3.zero);
                }
                else
                {
                    Inertialization.RightContactTransition(currentRightToesPosition, currentRightToesVelocity, currentRightToesPosition, currentRightToesVelocity);
                }
            }
            else if (unlockRightContact || (IsRightFootContact && !pose.RightFootContact))
            {
                IsRightFootContact = false;

                if (Inertialize)
                {
                    Inertialization.RightContactTransition(RightFootContact, float3.zero, currentRightToesPosition, currentRightToesVelocity);
                }
                else
                {
                    Inertialization.RightContactTransition(currentRightToesPosition, currentRightToesVelocity, currentRightToesPosition, currentRightToesVelocity);
                }
            }

            // IK to place the foot
            if (FootLock)
            {
                // Left Foot IK
                TwoJointIK.Solve((Vector3)leftContactPosition + (SkeletonTransforms[LeftFootIndex].position - SkeletonTransforms[LeftToesIndex].position),
                                 SkeletonTransforms[LeftUpperLegIndex],
                                 SkeletonTransforms[LeftLowerLegIndex],
                                 SkeletonTransforms[LeftFootIndex],
                                 LeftFootPoleContact);
                // Right Foot IK
                TwoJointIK.Solve((Vector3)rightContactPosition + (SkeletonTransforms[RightFootIndex].position - SkeletonTransforms[RightToesIndex].position),
                                 SkeletonTransforms[RightUpperLegIndex],
                                 SkeletonTransforms[RightLowerLegIndex],
                                 SkeletonTransforms[RightFootIndex],
                                 RightFootPoleContact);
            }
        }

        private float3 GetPositionLocalCharacter(float3 worldPosition)
        {
            return SkeletonTransforms[0].InverseTransformPoint(worldPosition);
        }

        private float3 GetDirectionLocalCharacter(float3 worldDir)
        {
            return SkeletonTransforms[0].InverseTransformDirection(worldDir);
        }

        /// <summary>
        /// Adds an offset to the current transform space (useful to move the character to a different position)
        /// Simply changing the transform won't work because motion matching applies root motion based on the current motion matching search space
        /// </summary>
        public void SetPosAdjustment(float3 posAdjustment)
        {
            MMTransformOriginPose += posAdjustment;
        }
        /// <summary>
        /// Adds a rot offset to the current transform space (useful to rotate the character to a different direction)
        /// Simply changing the transform won't work because motion matching applies root motion based on the current motion matching search space
        /// </summary>
        public void SetRotAdjustment(quaternion rotAdjustment)
        {
            MMTransformOriginRot = math.mul(rotAdjustment, MMTransformOriginRot);
        }

        public int GetCurrentFrame()
        {
            return CurrentFrame;
        }
        public int GetLastFrame()
        {
            return LastMMSearchFrame;
        }
        public void SetCurrentFrame(int frame)
        {
            CurrentFrame = frame;
        }
        public FeatureSet GetFeatureSet()
        {
            FeatureSet featureSet = FeatureSet;
            if (SquatIndex > 0) featureSet = SquatFeatureSets[SquatIndex - 1];
            return featureSet;
        }
        public NativeArray<float> GetQueryFeature()
        {
            return QueryFeature;
        }
        public NativeArray<float> UpdateAndGetFeatureWeights()
        {
            int offset = 0;
            for (int i = 0; i < MMData.TrajectoryFeatures.Count; i++)
            {
                TrajectoryFeature feature = MMData.TrajectoryFeatures[i];
                float weight = FeatureWeights[i] * Responsiveness;
                for (int p = 0; p < feature.FramesPrediction.Length; ++p)
                {
                    for (int f = 0; f < (feature.Project ? 2 : 3); f++)
                    {
                        FeaturesWeightsNativeArray[offset + f] = weight;
                    }
                    offset += (feature.Project ? 2 : 3);
                }
            }
            for (int i = 0; i < MMData.PoseFeatures.Count; i++)
            {
                PoseFeature feature = MMData.PoseFeatures[i];
                float weight = FeatureWeights[i + MMData.TrajectoryFeatures.Count] * Quality;
                FeaturesWeightsNativeArray[offset + 0] = weight;
                FeaturesWeightsNativeArray[offset + 1] = weight;
                FeaturesWeightsNativeArray[offset + 2] = weight;
                offset += 3;
            }
            return FeaturesWeightsNativeArray;
        }

        /// <summary>
        /// Returns the skeleton used by Motion Matching
        /// </summary>
        public Skeleton GetSkeleton()
        {
            return PoseSet.Skeleton;
        }

        /// <summary>
        /// Returns the transforms used by Motion Matching to simulate the skeleton
        /// </summary>
        public Transform[] GetSkeletonTransforms()
        {
            return SkeletonTransforms;
        }

        private void OnDestroy()
        {
            if (FeatureSet != null) FeatureSet.Dispose();
            for (int i = 0; i < SquatFeatureSets.Length; i++) if (SquatFeatureSets[i] != null) SquatFeatureSets[i].Dispose();
            if (QueryFeature != null && QueryFeature.IsCreated) QueryFeature.Dispose();
            if (SearchResult != null && SearchResult.IsCreated) SearchResult.Dispose();
            if (FeaturesWeightsNativeArray != null && FeaturesWeightsNativeArray.IsCreated) FeaturesWeightsNativeArray.Dispose();
        }

        private void OnApplicationQuit()
        {
            OnDestroy();
        }

        [System.Serializable]
        public struct SquatDataset
        {
            public MotionMatchingData MMData;
            public float PercentageHeight; // Below this percentage of height it will activate this squat dataset
        }

#if UNITY_EDITOR
        private void OnDrawGizmos()
        {
            // Skeleton
            if (SkeletonTransforms == null || PoseSet == null) return;

            if (DebugSkeleton)
            {
                Gizmos.color = Color.red;
                for (int i = 2; i < SkeletonTransforms.Length; i++) // skip Simulation Bone
                {
                    Transform t = SkeletonTransforms[i];
                    GizmosExtensions.DrawLine(t.parent.position, t.position, 3);
                }
            }

            // Character
            PoseSet poseSet = PoseSet;
            if (SquatIndex > 0) poseSet = SquatPoseSets[SquatIndex - 1];
            if (poseSet == null) return;

            int currentFrame = CurrentFrame;
            poseSet.GetPose(currentFrame, out PoseVector pose);
            float3 characterOrigin = SkeletonTransforms[0].position;
            float3 characterForward = SkeletonTransforms[0].forward;
            if (DebugCurrent)
            {
                Gizmos.color = new Color(1.0f, 0.0f, 0.5f, 1.0f);
                Gizmos.DrawSphere(characterOrigin, SpheresRadius);
                GizmosExtensions.DrawArrow(characterOrigin, characterOrigin + characterForward, thickness: 3);
            }
            if (DebugContacts)
            {
                Gizmos.color = Color.green;
                if (IsLeftFootContact)
                {
                    Gizmos.DrawSphere(SkeletonTransforms[LeftToesIndex].position, SpheresRadius);
                }
                if (IsRightFootContact)
                {
                    Gizmos.DrawSphere(SkeletonTransforms[RightToesIndex].position, SpheresRadius);
                }
            }

            // Feature Set
            FeatureSet featureSet = FeatureSet;
            if (SquatIndex > 0) featureSet = SquatFeatureSets[SquatIndex - 1];
            if (featureSet == null) return;

            FeatureDebug.DrawFeatureGizmos(featureSet, MMData, SpheresRadius, currentFrame, characterOrigin, characterForward,
                                           SkeletonTransforms, PoseSet.Skeleton, DebugPose, DebugTrajectory);
        }
#endif
    }
}