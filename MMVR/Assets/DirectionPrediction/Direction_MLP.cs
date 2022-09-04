using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

public class Direction_MLP
{
    private Model DirectionPredictorModel;
    private IWorker PoseWorker;

    public Direction_MLP(NNModel directionPredictorSource, int inputSize)
    {
        DirectionPredictorModel = ModelLoader.Load(directionPredictorSource);
        PoseWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharp, DirectionPredictorModel);
    }

    public void Predict(float[] inTrackerInfo, float[] outDirection)
    {
        // Pose Synthesizer --------------
        // Tensor
        Tensor input = new Tensor(1, 1, 1, inTrackerInfo.Length, inTrackerInfo);
        // Run
        PoseWorker.Execute(input);
        // Copy output
        Tensor DirectionOutput = PoseWorker.PeekOutput();
        for (int i = 0; i < outDirection.Length; ++i)
        {
            outDirection[i] = DirectionOutput[0, 0, 0, i];
        }

        // Dispose
        input.Dispose();
        DirectionOutput.Dispose();
    }
}
