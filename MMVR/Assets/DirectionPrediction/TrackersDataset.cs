using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MotionMatching;
using Unity.Mathematics;
using System.IO;
using System;

public class TrackersDataset
{
    public float3[,] Positions; // For Debug Only
    public float[,] Data; // Rows: One Pose, Colums: Features. Data = [(rot, vel, angVel) x NumberTrackers]

    public const int NumberTrackers = 3;
    public int NumberPoses { get; private set; }
    public int NumberFeaturesPerTracker { get; private set; }
    public int NumberFeatures { get; private set; }

    public const int RotDimensions = 6;
    public const int VelDimensions = 3;
    public const int AngVelDimensions = 3;

    public float[] Mean;
    public float[] StandardDeviation;

    private HumanBodyBones[] TrackersParents =
    {
        HumanBodyBones.Head,      // HMD
        HumanBodyBones.LeftHand,  // Left Controller
        HumanBodyBones.RightHand, // Right Controller
    };
    private float3[] TrackersLocalOffset = // Offset of the Trackers from its parent joint (relative to T-Pose)
    {
        new float3(0, 0, 0.1f),   // HMD
        new float3(0, 0, 0.175f), // Left Controller
        new float3(0, 0, 0.175f), // Right Controller
    };
    private Skeleton.Joint[] TrackersParentsJoints;

    public quaternion[] LocalToVRSpace =
    {
        quaternion.identity,            // HMD
        quaternion.Euler(0, -90.0f, 0), // Left Controller
        quaternion.Euler(0, 90.0f, 0),  // Right Controller  
    };
    public quaternion[] VRSpaceToTracker; // Rotation from VR space to tracker space

    private float3[] WorldPosCache;
    private quaternion[] WorldRotCache;
    private float3[] WorldVelCache;
    private float3[] WorldAngVelCache;

    private float3[] WorldHMD;
    private quaternion[] WorldProjectedDirHMD;
    private quaternion[] SimulationBoneRot;

    public TrackersDataset() { }

    public TrackersDataset(PoseSet poseSet, MotionSynthesisData msData)
    {
        NumberPoses = poseSet.NumberPoses;
        NumberFeaturesPerTracker = RotDimensions + VelDimensions + AngVelDimensions;
        NumberFeatures = NumberTrackers * NumberFeaturesPerTracker;

        WorldPosCache = new float3[poseSet.Skeleton.Joints.Count];
        WorldRotCache = new quaternion[poseSet.Skeleton.Joints.Count];
        WorldVelCache = new float3[poseSet.Skeleton.Joints.Count];
        WorldAngVelCache = new float3[poseSet.Skeleton.Joints.Count];

        WorldHMD = new float3[poseSet.NumberPoses];
        WorldProjectedDirHMD = new quaternion[poseSet.NumberPoses];
        SimulationBoneRot = new quaternion[poseSet.NumberPoses];

        TrackersParentsJoints = new Skeleton.Joint[NumberTrackers];
        for (int i = 0; i < NumberTrackers; i++)
        {
            if (!poseSet.Skeleton.Find(TrackersParents[i], out Skeleton.Joint joint))
            {
                Debug.LogError("TrackersDataset: Tracker " + i + " not found in Skeleton");
            }
            TrackersParentsJoints[i] = joint;
        }

        VRSpaceToTracker = new quaternion[NumberTrackers];

        // Create and Fill Dataset
        Data = new float[NumberPoses, NumberFeatures];
        Positions = new float3[NumberPoses, NumberTrackers];
        for (int row = 0; row < NumberPoses; row++)
        {
            poseSet.GetPose(row, out PoseVector pose);
            GetWorldInfo(poseSet.Skeleton, pose,
                         WorldPosCache, WorldRotCache, WorldVelCache, WorldAngVelCache);

            for (int t = 0; t < NumberTrackers; t++)
            {
                FillTrackerData(row, t, poseSet.Skeleton, msData);
            }
            SimulationBoneRot[row] = WorldRotCache[0];
        }

        Normalize();
    }

    // For debugging purposes
    public float3 GetSimulationBoneByHMD(int index)
    {
        float3 projection = WorldHMD[index];
        projection.y = 0;
        return projection;
    }
    public quaternion GetProjectedDirHMD(int index)
    {
        return WorldProjectedDirHMD[index];
    }

    public void GetTrackersInfo(int index, float[] data, bool denormalize = false)
    {
        for (int col = 0; col < NumberFeatures; col++)
        {
            float value = Data[index, col];
            if (denormalize)
            {
                value = value * StandardDeviation[col] + Mean[col];
            }
            data[col] = value;
        }
    }

    public float3 NormalizePosition(int trackerIndex, float3 position)
    {
        int offset = trackerIndex * NumberFeaturesPerTracker;
        float3 mean = new float3(Mean[offset], Mean[offset + 1], Mean[offset + 2]);
        float3 std = new float3(StandardDeviation[offset], StandardDeviation[offset + 1], StandardDeviation[offset + 2]);
        return (position - mean) / std;
    }

    public float3 DenormalizePosition(int trackerIndex, float3 position)
    {
        int offset = trackerIndex * NumberFeaturesPerTracker;
        float3 mean = new float3(Mean[offset], Mean[offset + 1], Mean[offset + 2]);
        float3 std = new float3(StandardDeviation[offset], StandardDeviation[offset + 1], StandardDeviation[offset + 2]);
        return position * std + mean;
    }

    public void Normalize(float[] data)
    {
        Debug.Assert(data.Length == NumberFeatures, "TrackersDataset: data.Length != NumberFeatures");
        for (int col = 0; col < NumberFeatures; col++)
        {
            data[col] = (data[col] - Mean[col]) / StandardDeviation[col];
        }
    }

    public void Denormalize(float[] data)
    {
        Debug.Assert(data.Length == NumberFeatures, "TrackersDataset: Denormalize: data.Length != NumberFeatures");
        for (int col = 0; col < NumberFeatures; col++)
        {
            data[col] = data[col] * StandardDeviation[col] + Mean[col];
        }
    }

    public void ChangeMeanAndStd(float[] mean, float[] std)
    {
        // Denormalize
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeatures; col++)
            {
                Data[row, col] = Data[row, col] * StandardDeviation[col] + Mean[col];
            }
        }
        Mean = mean;
        StandardDeviation = std;
        // Normalize
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeatures; col++)
            {
                Data[row, col] = (Data[row, col] - Mean[col]) / StandardDeviation[col];
            }
        }
    }

    private void FillTrackerData(int row, int t, Skeleton skeleton, MotionSynthesisData msData)
    {
        int offset = t * NumberFeaturesPerTracker;
        // Rotation from forward (0, 0, 1) to the forward computed in T-Pose
        float3 localForwardJoint = msData.GetLocalForward(TrackersParentsJoints[t].Index);
        float3 localForwardUp = math.mul(quaternion.Euler(math.radians(-90), 0, 0), localForwardJoint);
        quaternion forward = quaternion.LookRotation(localForwardJoint, localForwardUp);
        // Tracker Offset relative to its parent joint
        float3 trackerOffset = math.mul(forward, TrackersLocalOffset[t]);
        // World Tracker Position
        float3 pos = WorldPosCache[TrackersParentsJoints[t].Index] +
                     math.mul(WorldRotCache[TrackersParentsJoints[t].Index], trackerOffset);
        // World Tracker Rotation
        quaternion rot = math.mul(WorldRotCache[TrackersParentsJoints[t].Index], forward);
        if (t == 0)
        {
            WorldHMD[row] = pos;
            float3 projectedDir = math.mul(rot, math.forward());
            projectedDir.y = 0;
            projectedDir = math.normalize(projectedDir);
            WorldProjectedDirHMD[row] = quaternion.LookRotation(projectedDir, math.up());
        }
        // World Tracker Velocity
        // Read comment (velocity computation) in GetWorldInfo(...)
        float3 vel = WorldVelCache[TrackersParentsJoints[t].Index] +
                     math.cross(WorldAngVelCache[TrackersParentsJoints[t].Index], trackerOffset);
        // World Tracker Angular Velocity
        float3 angVel = WorldAngVelCache[TrackersParentsJoints[t].Index];

        // Change to Character Space (Simulation Bone is HMD projected on the ground)
        float3 hmdPos = WorldHMD[row];
        hmdPos.y = 0;
        float3 simulationBonePosition = hmdPos;
        quaternion invCharacterRot = math.inverse(WorldRotCache[0]);
        pos = math.mul(invCharacterRot, pos - simulationBonePosition);
        quaternion invHMDRot = math.inverse(WorldProjectedDirHMD[row]);
        rot = math.mul(invHMDRot, rot);
        vel = math.mul(invHMDRot, vel);
        angVel = math.mul(invHMDRot, angVel);

        if (row == 0)
        {
            float3 localForwardSBTPose = msData.GetLocalForward(1); // hips
            localForwardSBTPose.y = 0.0f;
            localForwardSBTPose = math.normalize(localForwardSBTPose);
            float3 localForwardUpSBTPose = math.up();
            quaternion forwardSBTPose = quaternion.LookRotation(localForwardSBTPose, localForwardUpSBTPose);
            // First time compute VR to Tracker space data
            VRSpaceToTracker[t] = math.mul(math.inverse(forwardSBTPose), forward);
        }

        Positions[row, t] = pos;
        // Convert to 2-axis representation
        float3x2 rotContinuous = MathExtensions.QuaternionToContinuous(rot);
        Data[row, offset + 0] = rotContinuous.c0[0];
        Data[row, offset + 1] = rotContinuous.c0[1];
        Data[row, offset + 2] = rotContinuous.c0[2];
        Data[row, offset + 3] = rotContinuous.c1[0];
        Data[row, offset + 4] = rotContinuous.c1[1];
        Data[row, offset + 5] = rotContinuous.c1[2];
        Data[row, offset + 6] = vel.x;
        Data[row, offset + 7] = vel.y;
        Data[row, offset + 8] = vel.z;
        Data[row, offset + 9] = angVel.x;
        Data[row, offset + 10] = angVel.y;
        Data[row, offset + 11] = angVel.z;
    }

    private void GetWorldInfo(Skeleton skeleton, PoseVector pose,
                              float3[] outWorldPos, quaternion[] outWorldRot, float3[] outWorldVel, float3[] outWorldAngVel)
    {
        outWorldPos[0] = pose.JointLocalPositions[0];
        outWorldRot[0] = pose.JointLocalRotations[0];
        outWorldVel[0] = pose.JointLocalVelocities[0];
        outWorldAngVel[0] = pose.JointLocalAngularVelocities[0];
        for (int j = 1; j < skeleton.Joints.Count; ++j)
        {
            Skeleton.Joint joint = skeleton.Joints[j];
            float3 rotatedLocalOffset = math.mul(outWorldRot[joint.ParentIndex], pose.JointLocalPositions[j]);
            outWorldPos[j] = rotatedLocalOffset + outWorldPos[joint.ParentIndex];
            outWorldRot[j] = math.mul(outWorldRot[joint.ParentIndex], pose.JointLocalRotations[j]);
            // Given a fixed point 'O', a point 'A' relative to 'O', and the angular velocity 'w' of 'O'
            // the velocity 'V' of 'A' is 'V = w x OA' where 'x' is the cross product and 'OA' is the vector from 'O' to 'A'
            // Here, we add the local velocity + the velocity caused by the angular velocity + parent velocity
            outWorldVel[j] = math.mul(outWorldRot[joint.ParentIndex], pose.JointLocalVelocities[j]) +
                             math.cross(outWorldAngVel[joint.ParentIndex], rotatedLocalOffset) +
                             outWorldVel[joint.ParentIndex];
            outWorldAngVel[j] = math.mul(outWorldRot[joint.ParentIndex], pose.JointLocalAngularVelocities[j]) +
                                outWorldAngVel[joint.ParentIndex];
        }
    }

    private void Normalize()
    {
        ComputeMeanAndStandardDeviation();
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeatures; col++)
            {
                Data[row, col] = (Data[row, col] - Mean[col]) / StandardDeviation[col];
            }
        }
    }

    private void ComputeMeanAndStandardDeviation()
    {
        // Mean for each dimension
        Mean = new float[NumberFeatures];
        // Variance for each dimension
        Span<float> variance = stackalloc float[NumberFeatures];
        // Standard Deviation for each dimension
        StandardDeviation = new float[NumberFeatures];

        // Compute Means for each dimension of each feature
        for (int col = 0; col < NumberFeatures; col++)
        {
            for (int row = 0; row < NumberPoses; row++)
            {
                Mean[col] += Data[row, col];
            }
        }
        for (int col = 0; col < NumberFeatures; col++)
        {
            Mean[col] /= NumberPoses;
        }
        // Compute Variance for each dimension of each feature - variance = (x - mean)^2 / n
        for (int col = 0; col < NumberFeatures; col++)
        {
            for (int row = 0; row < NumberPoses; row++)
            {
                float diff = Data[row, col] - Mean[col];
                variance[col] += diff * diff;
            }
        }
        for (int col = 0; col < NumberFeatures; col++)
        {
            variance[col] /= NumberPoses;
            // Debug.Assert(variance[j] > 0, "Variance is zero, feature with no variation is probably a bug, j: " + j + " TotalNumberFeatures: " + NumberFeatures);
        }

        // Compute Standard Deviations of a feature as the average std across all dimensions - std = sqrt(variance)
        const int rotDim = RotDimensions;
        const int velDim = VelDimensions;
        const int angVelDim = AngVelDimensions;
        for (int t = 0; t < NumberTrackers; t++)
        {
            int offset = t * NumberFeaturesPerTracker;
            // Rot
            float std = 0;
            for (int col = 0; col < rotDim; col++)
            {
                std += math.sqrt(variance[offset + col]);
            }
            std /= rotDim;
            Debug.Assert(std > 0, "Standard deviation is zero, feature with no variation is probably a bug");
            for (int col = 0; col < rotDim; col++)
            {
                StandardDeviation[offset + col] = std;
            }
            offset += rotDim;
            // Vel
            std = 0;
            for (int col = 0; col < velDim; col++)
            {
                std += math.sqrt(variance[offset + col]);
            }
            std /= velDim;
            Debug.Assert(std > 0, "Standard deviation is zero, feature with no variation is probably a bug");
            for (int col = 0; col < velDim; col++)
            {
                StandardDeviation[offset + col] = std;
            }
            offset += velDim;
            // AngVel
            std = 0;
            for (int col = 0; col < angVelDim; col++)
            {
                std += math.sqrt(variance[offset + col]);
            }
            std /= angVelDim;
            Debug.Assert(std > 0, "Standard deviation is zero, feature with no variation is probably a bug");
            for (int col = 0; col < angVelDim; col++)
            {
                StandardDeviation[offset + col] = std;
            }
            offset += angVelDim;
        }
    }


    public void Serialize(string path, string fileName)
    {
        Directory.CreateDirectory(path); // create directory and parent directories if they don't exist

        using (var stream = File.Open(Path.Combine(path, fileName + ".mstrackers"), FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8))
            {
                // Header
                writer.Write((uint)NumberPoses);
                writer.Write((uint)NumberTrackers);
                writer.Write((uint)NumberFeaturesPerTracker);
                writer.Write((uint)NumberFeatures);
                // Mean and Standard Deviation
                for (int i = 0; i < NumberFeatures; i++)
                {
                    writer.Write(Mean[i]);
                    writer.Write(StandardDeviation[i]);
                }
                // Poses
                for (int i = 0; i < NumberPoses; i++)
                {
                    for (int j = 0; j < NumberFeatures; j++)
                    {
                        writer.Write(Data[i, j]);
                    }
                }
                // Positions
                for (int i = 0; i < NumberPoses; i++)
                {
                    for (int t = 0; t < NumberTrackers; t++)
                    {
                        writer.Write(Positions[i, t].x);
                        writer.Write(Positions[i, t].y);
                        writer.Write(Positions[i, t].z);
                    }
                }
                // VRSpaceToTracker
                for (int t = 0; t < NumberTrackers; t++)
                {
                    writer.Write(VRSpaceToTracker[t].value.x);
                    writer.Write(VRSpaceToTracker[t].value.y);
                    writer.Write(VRSpaceToTracker[t].value.z);
                    writer.Write(VRSpaceToTracker[t].value.w);
                }
            }
        }

        using (var stream = File.Open(Path.Combine(path, fileName + ".mstrackersdebug"), FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8))
            {
                // Header
                writer.Write((uint)NumberPoses);
                // HMD Debug
                for (int i = 0; i < NumberPoses; i++)
                {
                    writer.Write(WorldHMD[i].x);
                    writer.Write(WorldHMD[i].y);
                    writer.Write(WorldHMD[i].z);
                    writer.Write(WorldProjectedDirHMD[i].value.x);
                    writer.Write(WorldProjectedDirHMD[i].value.y);
                    writer.Write(WorldProjectedDirHMD[i].value.z);
                    writer.Write(WorldProjectedDirHMD[i].value.w);
                }
            }
        }
    }

    public bool Deserialize(string path, string fileName)
    {
        string trackersPath = Path.Combine(path, fileName + ".mstrackers");
        if (File.Exists(trackersPath))
        {
            using (var stream = File.Open(trackersPath, FileMode.Open))
            {
                using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8))
                {
                    // Header
                    NumberPoses = (int)reader.ReadUInt32();
                    int numberTrackers = (int)reader.ReadUInt32();
                    if (numberTrackers != NumberTrackers)
                    {
                        Debug.LogError("TrackersDataset: Number of trackers in file does not match number of trackers in dataset");
                    }
                    NumberFeaturesPerTracker = (int)reader.ReadUInt32();
                    NumberFeatures = (int)reader.ReadUInt32();
                    // Mean and Standard Deviation
                    Mean = new float[NumberFeatures];
                    StandardDeviation = new float[NumberFeatures];
                    for (int i = 0; i < NumberFeatures; i++)
                    {
                        Mean[i] = reader.ReadSingle();
                        StandardDeviation[i] = reader.ReadSingle();
                    }
                    // Poses
                    Data = new float[NumberPoses, NumberFeatures];
                    for (int i = 0; i < NumberPoses; i++)
                    {
                        for (int j = 0; j < NumberFeatures; j++)
                        {
                            Data[i, j] = reader.ReadSingle();
                        }
                    }
                    // Positions
                    Positions = new float3[NumberPoses, NumberTrackers];
                    for (int i = 0; i < NumberPoses; i++)
                    {
                        for (int t = 0; t < NumberTrackers; t++)
                        {
                            Positions[i, t].x = reader.ReadSingle();
                            Positions[i, t].y = reader.ReadSingle();
                            Positions[i, t].z = reader.ReadSingle();
                        }
                    }
                    // VRSpaceToTracker
                    VRSpaceToTracker = new quaternion[NumberTrackers];
                    for (int t = 0; t < NumberTrackers; t++)
                    {
                        VRSpaceToTracker[t].value.x = reader.ReadSingle();
                        VRSpaceToTracker[t].value.y = reader.ReadSingle();
                        VRSpaceToTracker[t].value.z = reader.ReadSingle();
                        VRSpaceToTracker[t].value.w = reader.ReadSingle();
                    }
                }
            }
        }
        else return false;

        trackersPath = Path.Combine(path, fileName + ".mstrackersdebug");
        if (File.Exists(trackersPath))
        {
            using (var stream = File.Open(trackersPath, FileMode.Open))
            {
                using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8))
                {
                    // Header
                    NumberPoses = (int)reader.ReadUInt32();
                    // HMD Debug
                    WorldHMD = new float3[NumberPoses];
                    WorldProjectedDirHMD = new quaternion[NumberPoses];
                    for (int i = 0; i < NumberPoses; i++)
                    {
                        WorldHMD[i] = new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                        WorldProjectedDirHMD[i] = new quaternion(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                    }
                }
            }
        }
        else return false;

        return true;
    }

    public static bool DeserializeMeanAndStd(string path, string fileName, out float[] mean, out float[] std)
    {
        mean = null;
        std = null;
        string posePath = Path.Combine(path, fileName + ".mstrackers");
        if (File.Exists(posePath))
        {
            using (var stream = File.Open(posePath, FileMode.Open))
            {
                using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8))
                {
                    // Header
                    int numberPoses = (int)reader.ReadUInt32();
                    int numberTrackers = (int)reader.ReadUInt32();
                    if (numberTrackers != NumberTrackers)
                    {
                        Debug.LogError("TrackersDataset: Number of trackers in file does not match number of trackers in dataset");
                    }
                    int numberFeaturesPerTracker = (int)reader.ReadUInt32();
                    int numberFeatures = (int)reader.ReadUInt32();
                    // Mean and Standard Deviation
                    mean = new float[numberFeatures];
                    std = new float[numberFeatures];
                    for (int i = 0; i < numberFeatures; i++)
                    {
                        mean[i] = reader.ReadSingle();
                        std[i] = reader.ReadSingle();
                    }
                }
            }
        }
        else return false;

        return true;
    }
}
