using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using System;

namespace MotionMatching
{
    [RequireComponent(typeof(Animator))]
    public class MotionMatchingSkinnedMeshRenderer : MonoBehaviour
    {
        public MotionMatchingController MotionMatching;

        private Animator Animator;

        // Retargeting
        // Initial orientations of the bones The code assumes the initial orientations are in T-Pose
        private Quaternion[] SourceTPose;
        private Quaternion[] TargetTPose;
        // Mapping from BodyJoints to the actual transforms
        private Transform[] SourceBones;
        private Transform[] TargetBones;
        public bool ShouldRetarget { get { return MotionMatching.MMData.BVHTPose != null; } }

        private void Awake()
        {
            Animator = GetComponent<Animator>();
        }

        private void OnEnable()
        {
            MotionMatching.OnSkeletonTransformUpdated += OnSkeletonTransformUpdated;
        }

        private void OnDisable()
        {
            MotionMatching.OnSkeletonTransformUpdated -= OnSkeletonTransformUpdated;
        }

        private void Start()
        {
            // BindSkinnedMeshRenderers();
            if (ShouldRetarget) InitRetargeting();
        }

        private void InitRetargeting()
        {
            MotionMatchingData mmData = MotionMatching.MMData;
            SourceTPose = new Quaternion[BodyJoints.Length];
            TargetTPose = new Quaternion[BodyJoints.Length];
            SourceBones = new Transform[BodyJoints.Length];
            TargetBones = new Transform[BodyJoints.Length];
            // Source TPose (BVH with TPose)
            BVHImporter bvhImporter = new BVHImporter();
            // Animation containing in the first frame a TPose
            BVHAnimation tposeAnimation = bvhImporter.Import(mmData.BVHTPose, mmData.UnitScale, true);
            // Store Rotations
            // Source
            Skeleton skeleton = tposeAnimation.Skeleton;
            for (int i = 0; i < BodyJoints.Length; i++)
            {
                if (mmData.GetJointName(BodyJoints[i], out string jointName) &&
                    skeleton.Find(jointName, out Skeleton.Joint joint))
                {
                    // Get the rotation for the first frame of the animation
                    SourceTPose[i] = tposeAnimation.GetWorldRotation(joint, 0);
                }
            }
            // Target
            Quaternion avatarRot = Animator.transform.rotation;
            SkeletonBone[] targetSkeletonBones = Animator.avatar.humanDescription.skeleton;
            Quaternion hipsRot = avatarRot;
            for (int i = 0; i < BodyJoints.Length; i++)
            {
                Transform targetJoint = Animator.GetBoneTransform(BodyJoints[i]);

                // Use Array.FindIndex to find the index of the joint in the targetSkeletonBones array
                int targetJointIndex = Array.FindIndex(targetSkeletonBones, bone => bone.name == targetJoint.name);
                Debug.Assert(targetJointIndex != -1, "Target joint not found: " + targetJoint.name);

                // Initialize the rotation as the local rotation of the joint
                Quaternion cumulativeRotation = targetSkeletonBones[targetJointIndex].rotation;

                // Traverse up the hierarchy until reaching the Animator's transform
                Transform currentTransform = targetJoint.parent;
                while (currentTransform != null && currentTransform != Animator.transform)
                {
                    int parentIndex = Array.FindIndex(targetSkeletonBones, bone => bone.name == currentTransform.name);
                    if (parentIndex != -1)
                    {
                        // Multiply with the parent's local rotation
                        cumulativeRotation = targetSkeletonBones[parentIndex].rotation * cumulativeRotation;
                    }
                    Debug.Assert(parentIndex != -1, "Parent joint not found: " + currentTransform.name);

                    // Move to the next parent in the hierarchy
                    currentTransform = currentTransform.parent;
                }

                // Store the world rotation
                TargetTPose[i] = cumulativeRotation;
                if (BodyJoints[i] == HumanBodyBones.Hips)
                {
                    hipsRot = math.mul(avatarRot, cumulativeRotation);
                }
            }
            // Correct rotations so they are facing the same direction as the target
            // Correct Source
            float3 currentDirection = math.mul(SourceTPose[0], mmData.HipsForwardLocalVector);
            currentDirection.y = 0;
            currentDirection = math.normalize(currentDirection);
            float3 forwardLocalVector = math.mul(math.inverse(hipsRot), math.forward());
            float3 targetDirection = transform.TransformDirection(forwardLocalVector);
            targetDirection.y = 0;
            targetDirection = math.normalize(targetDirection);
            quaternion correctionRot = MathExtensions.FromToRotation(currentDirection, targetDirection, new float3(0, 1, 0));
            for (int i = 0; i < BodyJoints.Length; i++)
            {
                SourceTPose[i] = math.mul(correctionRot, SourceTPose[i]);
            }
            // Store Transforms
            Transform[] mmBones = MotionMatching.GetSkeletonTransforms();
            Dictionary<string, Transform> boneDict = new Dictionary<string, Transform>();
            foreach (Transform bone in mmBones)
            {
                boneDict.Add(bone.name, bone);
            }
            // Source
            for (int i = 0; i < BodyJoints.Length; i++)
            {
                if (mmData.GetJointName(BodyJoints[i], out string jointName) &&
                    boneDict.TryGetValue(jointName, out Transform bone))
                {
                    SourceBones[i] = bone;
                }
            }
            // Target
            for (int i = 0; i < BodyJoints.Length; i++)
            {
                TargetBones[i] = Animator.GetBoneTransform(BodyJoints[i]);
            }
        }

        private void OnSkeletonTransformUpdated()
        {
            if (!ShouldRetarget) return;
            // Motion
            transform.position = MotionMatching.transform.position;
            // Retargeting
            for (int i = 0; i < BodyJoints.Length; i++)
            {
                Quaternion sourceTPoseRotation = SourceTPose[i];
                Quaternion targetTPoseRotation = TargetTPose[i];
                Quaternion sourceRotation = SourceBones[i].rotation;
                /*
                    R_t = Rotation transforming from target local space to world space
                    R_s = Rotation transforming from source local space to world space
                    R_t = R_s * R_st (R_st is a matrix transforming from target to source space)
                    // It makes sense because R_st will be mapping from target to source, and R_s from source to world.
                    // The result is transforming from T to world, which is what R_t does.
                    RTPose_t = RTPose_s * R_st
                    R_st = (RTPose_s)^-1 * RTPose_t
                    R_t = R_s * (R_st)^-1 * RTPose_t
                */
                TargetBones[i].rotation = sourceRotation * Quaternion.Inverse(sourceTPoseRotation) * targetTPoseRotation;
            }
            // Hips Height
            TargetBones[0].position = MotionMatching.GetSkeletonTransforms()[1].position;
            // Correct Toe if under ground
            Transform leftToes = Animator.GetBoneTransform(HumanBodyBones.LeftToes);
            if (leftToes.position.y < 0.0f)
            {
                Transform leftFoot = Animator.GetBoneTransform(HumanBodyBones.LeftFoot);
                CorrectToes(leftToes, leftFoot);
            }
            Transform rightToes = Animator.GetBoneTransform(HumanBodyBones.RightToes);
            if (rightToes.position.y < 0.0f)
            {
                Transform rightFoot = Animator.GetBoneTransform(HumanBodyBones.RightFoot);
                CorrectToes(rightToes, rightFoot);
            }
        }

        private void CorrectToes(Transform toesT, Transform footT)
        {
            float3 toes = toesT.position;
            float3 foot = footT.position;
            float3 toesFoot = math.normalize(toes - foot);
            float3 desiredToesFoot = math.normalize(new float3(toes.x, 0.0f, toes.z) - foot);
            float angleCorrection = math.acos(math.clamp(math.dot(desiredToesFoot, toesFoot), -1.0f, 1.0f));
            float3 axisCorrection = math.normalize(math.cross(toesFoot, desiredToesFoot));
            quaternion rotCorrection = quaternion.AxisAngle(axisCorrection, angleCorrection);
            footT.rotation = math.mul(rotCorrection, footT.rotation);
        }

        // Used for retargeting. First parent, then children
        private HumanBodyBones[] BodyJoints =
        {
            HumanBodyBones.Hips,

            HumanBodyBones.Spine,
            HumanBodyBones.Chest,
            HumanBodyBones.UpperChest,

            HumanBodyBones.Neck,
            HumanBodyBones.Head,

            HumanBodyBones.LeftShoulder,
            HumanBodyBones.LeftUpperArm,
            HumanBodyBones.LeftLowerArm,
            HumanBodyBones.LeftHand,

            HumanBodyBones.RightShoulder,
            HumanBodyBones.RightUpperArm,
            HumanBodyBones.RightLowerArm,
            HumanBodyBones.RightHand,

            HumanBodyBones.LeftUpperLeg,
            HumanBodyBones.LeftLowerLeg,
            HumanBodyBones.LeftFoot,
            HumanBodyBones.LeftToes,

            HumanBodyBones.RightUpperLeg,
            HumanBodyBones.RightLowerLeg,
            HumanBodyBones.RightFoot,
            HumanBodyBones.RightToes
        };
    }
}