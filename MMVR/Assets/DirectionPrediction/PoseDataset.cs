using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MotionMatching;
using System;
using Unity.Mathematics;
using System.IO;

public class PoseDataset
{
    public float[,] Poses; // Rows: One Pose, Colums: Features. Poses = [jointRotations]
    public float[,] Hips; // Rows: One Pose, Colums: Features. Poses = [hipsPosition]

    public int NumberJoints { get; private set; } // Number of Joints in the Skeleton with Simulation Bone
    public int NumberPoses { get; private set; }
    public int NumberFeaturesPose { get; private set; }
    public int NumberFeaturesHips { get; private set; }

    public const int JointRotDimension = 6;
    public const int HipsPosDimension = 3;

    private float3[] JointLocalOffsets;

    private float[] Mean;
    private float[] StandardDeviation;

    public PoseDataset() { }

    public PoseDataset(PoseSet poseSet, TrackersDataset trackersDataset)
    {
        NumberJoints = poseSet.Skeleton.Joints.Count;
        NumberPoses = poseSet.NumberPoses;
        NumberFeaturesPose = NumberJoints * JointRotDimension;
        NumberFeaturesHips = HipsPosDimension;

        // Fill Joint Local Offsets
        JointLocalOffsets = new float3[NumberJoints];
        for (int i = 2; i < NumberJoints; i++) // Skip root and hips
        {
            JointLocalOffsets[i] = poseSet.Skeleton.Joints[i].LocalOffset;
        }

        // Create and Fill Dataset
        Poses = new float[NumberPoses, NumberFeaturesPose];
        Hips = new float[NumberPoses, NumberFeaturesHips];
        for (int row = 0; row < NumberPoses; row++)
        {
            poseSet.GetPose(row, out PoseVector pose);
            for (int col = 0; col < NumberJoints; col++)
            {
                int offsetPose = col * JointRotDimension;
                // Rotation
                // convert to 2-axis representation
                float3x2 rot = new float3x2();
                if (col == 0)
                {
                    // Simulation Bone w.r.t HMD
                    quaternion hmdProjDir = trackersDataset.GetProjectedDirHMD(row);
                    rot = MathExtensions.QuaternionToContinuous(math.mul(math.inverse(hmdProjDir), pose.JointLocalRotations[0]));
                }
                else
                {
                    rot = MathExtensions.QuaternionToContinuous(pose.JointLocalRotations[col]);
                }
                Poses[row, offsetPose + 0] = rot.c0[0];
                Poses[row, offsetPose + 1] = rot.c0[1];
                Poses[row, offsetPose + 2] = rot.c0[2];
                Poses[row, offsetPose + 3] = rot.c1[0];
                Poses[row, offsetPose + 4] = rot.c1[1];
                Poses[row, offsetPose + 5] = rot.c1[2];
            }
            // Hips Position
            // HMD projection defines the Simulation Bone... so we need to adjust the hips position
            float3 hmdSimulationBone = trackersDataset.GetSimulationBoneByHMD(row);
            float3 currentSimulationBone = pose.JointLocalPositions[0];
            float3 currentHipsPosition = currentSimulationBone + math.mul(pose.JointLocalRotations[0], pose.JointLocalPositions[1]);
            float3 newHipsPosition = math.mul(math.inverse(pose.JointLocalRotations[0]), currentHipsPosition - hmdSimulationBone);
            Hips[row, 0] = newHipsPosition.x;
            Hips[row, 1] = newHipsPosition.y;
            Hips[row, 2] = newHipsPosition.z;
        }

        Normalize();
    }

    public void GetPose(int index, float[] pose, float[] hips, bool denormalize = false)
    {
        for (int col = 0; col < NumberFeaturesPose; col++)
        {
            float value = Poses[index, col];
            if (denormalize)
            {
                value = value * StandardDeviation[col] + Mean[col];
            }
            pose[col] = value;
        }
        for (int col = 0; col < NumberFeaturesHips; col++)
        {
            float value = Hips[index, col];
            if (denormalize)
            {
                value = value * StandardDeviation[col + NumberFeaturesPose] + Mean[col + NumberFeaturesPose];
            }
            hips[col] = value;
        }
    }

    public void Denormalize(float[] pose, float[] hips)
    {
        Debug.Assert(pose.Length == NumberFeaturesPose, "Pose length is not correct");
        Debug.Assert(hips.Length == NumberFeaturesHips, "Hips length is not correct");
        for (int col = 0; col < NumberFeaturesPose; col++)
        {
            pose[col] = pose[col] * StandardDeviation[col] + Mean[col];
        }
        for (int col = 0; col < NumberFeaturesHips; col++)
        {
            hips[col] = hips[col] * StandardDeviation[col + NumberFeaturesPose] + Mean[col + NumberFeaturesPose];
        }
    }

    public void NormalizeSimulationBoneRot(float[] sbRot)
    {
        Debug.Assert(sbRot.Length == JointRotDimension, "Simulation Bone Rotation length is not correct");
        for (int col = 0; col < JointRotDimension; col++)
        {
            sbRot[col] = (sbRot[col] - Mean[col]) / StandardDeviation[col];
        }
    }

    public void DenormalizeSimulationBoneRot(float[] sbRot)
    {
        Debug.Assert(sbRot.Length == JointRotDimension, "Simulation Bone Rotation length is not correct");
        for (int col = 0; col < JointRotDimension; col++)
        {
            sbRot[col] = sbRot[col] * StandardDeviation[col] + Mean[col];
        }
    }

    public void ChangeMeanAndStd(float[] mean, float[] std)
    {
        // Denormalize
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesPose; col++)
            {
                Poses[row, col] = Poses[row, col] * StandardDeviation[col] + Mean[col];
            }
        }
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesHips; col++)
            {
                Hips[row, col] = Hips[row, col] * StandardDeviation[col + NumberFeaturesPose] + Mean[col + NumberFeaturesPose];
            }
        }
        Mean = mean;
        StandardDeviation = std;
        // Normalize
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesPose; col++)
            {
                Poses[row, col] = (Poses[row, col] - Mean[col]) / StandardDeviation[col];
            }
        }
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesHips; col++)
            {
                Hips[row, col] = (Hips[row, col] - Mean[col + NumberFeaturesPose]) / StandardDeviation[col + NumberFeaturesPose];
            }
        }
    }

    private void Normalize()
    {
        ComputeMeanAndStandardDeviation();
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesPose; col++)
            {
                Poses[row, col] = (Poses[row, col] - Mean[col]) / StandardDeviation[col];
            }
            for (int col = 0; col < NumberFeaturesHips; col++)
            {
                Hips[row, col] = (Hips[row, col] - Mean[col + NumberFeaturesPose]) / StandardDeviation[col + NumberFeaturesPose];
            }
        }
    }

    private void ComputeMeanAndStandardDeviation()
    {
        int numberFeatures = NumberFeaturesPose + NumberFeaturesHips;
        // Mean for each dimension
        Mean = new float[numberFeatures];
        // Variance for each dimension
        Span<float> variance = stackalloc float[numberFeatures];
        // Standard Deviation for each dimension
        StandardDeviation = new float[numberFeatures];

        // Compute Means for each dimension of each feature
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesPose; col++)
            {
                Mean[col] += Poses[row, col];
            }
            for (int col = 0; col < NumberFeaturesHips; col++)
            {
                Mean[col + NumberFeaturesPose] += Hips[row, col];
            }
        }
        for (int col = 0; col < numberFeatures; col++)
        {
            Mean[col] /= NumberPoses;
        }
        // Compute Variance for each dimension of each feature - variance = (x - mean)^2 / n
        for (int row = 0; row < NumberPoses; row++)
        {
            for (int col = 0; col < NumberFeaturesPose; col++)
            {
                float diff = Poses[row, col] - Mean[col];
                variance[col] += diff * diff;
            }
            for (int col = 0; col < NumberFeaturesHips; col++)
            {
                float diff = Hips[row, col] - Mean[col + NumberFeaturesPose];
                variance[col + NumberFeaturesPose] += diff * diff;
            }
        }
        for (int col = 0; col < numberFeatures; col++)
        {
            variance[col] /= NumberPoses;
            // Debug.Assert(variance[j] > 0, "Variance is zero, feature with no variation is probably a bug, j: " + j + " TotalNumberFeatures: " + NumberFeatures);
        }

        // Compute Standard Deviations of a feature as the average std across all dimensions - std = sqrt(variance)
        // Joint Rotations
        for (int joint = 0; joint < NumberJoints; joint++)
        {
            int offset = joint * JointRotDimension;
            float std = 0;
            for (int col = 0; col < JointRotDimension; col++)
            {
                std += math.sqrt(variance[offset + col]);
            }
            std /= JointRotDimension;
            Debug.Assert(std > 0, "Standard deviation is zero, feature with no variation is probably a bug");
            for (int col = 0; col < JointRotDimension; col++)
            {
                StandardDeviation[offset + col] = std;
            }
        }
        // Hips Position
        {
            int offset = NumberJoints * JointRotDimension;
            float std = 0;
            for (int col = 0; col < HipsPosDimension; col++)
            {
                std += math.sqrt(variance[offset + col]);
            }
            std /= HipsPosDimension;
            Debug.Assert(std > 0, "Standard deviation is zero, feature with no variation is probably a bug");
            for (int col = 0; col < HipsPosDimension; col++)
            {
                StandardDeviation[offset + col] = std;
            }
        }
    }

    public void Serialize(string path, string fileName)
    {
        Directory.CreateDirectory(path); // create directory and parent directories if they don't exist

        using (var stream = File.Open(Path.Combine(path, fileName + ".mspose"), FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8))
            {
                // Header
                writer.Write((uint)NumberPoses);
                writer.Write((uint)NumberFeaturesPose);
                writer.Write((uint)NumberFeaturesHips);
                writer.Write((uint)NumberJoints);
                // Mean and Standard Deviation
                for (int i = 0; i < (NumberFeaturesPose + NumberFeaturesHips); i++)
                {
                    writer.Write(Mean[i]);
                    writer.Write(StandardDeviation[i]);
                }
                // JointLocalOffsets
                for (int i = 0; i < NumberJoints; i++)
                {
                    writer.Write(JointLocalOffsets[i].x);
                    writer.Write(JointLocalOffsets[i].y);
                    writer.Write(JointLocalOffsets[i].z);
                }
                // Poses
                for (int i = 0; i < NumberPoses; i++)
                {
                    for (int j = 0; j < NumberFeaturesPose; j++)
                    {
                        writer.Write(Poses[i, j]);
                    }
                }
                // Hips
                for (int i = 0; i < NumberPoses; i++)
                {
                    for (int j = 0; j < NumberFeaturesHips; j++)
                    {
                        writer.Write(Hips[i, j]);
                    }
                }
            }
        }
    }

    public bool Deserialize(string path, string fileName)
    {
        string posePath = Path.Combine(path, fileName + ".mspose");
        if (File.Exists(posePath))
        {
            using (var stream = File.Open(posePath, FileMode.Open))
            {
                using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8))
                {
                    // Header
                    NumberPoses = (int)reader.ReadUInt32();
                    NumberFeaturesPose = (int)reader.ReadUInt32();
                    NumberFeaturesHips = (int)reader.ReadUInt32();
                    NumberJoints = (int)reader.ReadUInt32();
                    // Mean and Standard Deviation
                    int numberFeatures = NumberFeaturesPose + NumberFeaturesHips;
                    Mean = new float[numberFeatures];
                    StandardDeviation = new float[numberFeatures];
                    for (int i = 0; i < numberFeatures; i++)
                    {
                        Mean[i] = reader.ReadSingle();
                        StandardDeviation[i] = reader.ReadSingle();
                    }
                    // JointLocalOffsets
                    JointLocalOffsets = new float3[NumberJoints];
                    for (int i = 0; i < NumberJoints; i++)
                    {
                        JointLocalOffsets[i].x = reader.ReadSingle();
                        JointLocalOffsets[i].y = reader.ReadSingle();
                        JointLocalOffsets[i].z = reader.ReadSingle();
                    }
                    // Poses
                    Poses = new float[NumberPoses, NumberFeaturesPose];
                    for (int i = 0; i < NumberPoses; i++)
                    {
                        for (int j = 0; j < NumberFeaturesPose; j++)
                        {
                            Poses[i, j] = reader.ReadSingle();
                        }
                    }
                    // Hips
                    Hips = new float[NumberPoses, NumberFeaturesHips];
                    for (int i = 0; i < NumberPoses; i++)
                    {
                        for (int j = 0; j < NumberFeaturesHips; j++)
                        {
                            Hips[i, j] = reader.ReadSingle();
                        }
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
        string posePath = Path.Combine(path, fileName + ".mspose");
        if (File.Exists(posePath))
        {
            using (var stream = File.Open(posePath, FileMode.Open))
            {
                using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8))
                {
                    // Header
                    int numberPoses = (int)reader.ReadUInt32();
                    int numberFeaturesPose = (int)reader.ReadUInt32();
                    int numberFeaturesHips = (int)reader.ReadUInt32();
                    int numberJoints = (int)reader.ReadUInt32();
                    // Mean and Standard Deviation
                    int numberFeatures = numberFeaturesPose + numberFeaturesHips;
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
