using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
    using TrajectoryFeature = MotionMatchingData.TrajectoryFeature;

    public abstract class MotionMatchingCharacterController : MonoBehaviour
    {
        public event Action<float> OnUpdated;
        public event Action OnInputChangedQuickly;

        public MotionMatchingController SimulationBone; // MotionMatchingController's transform is the SimulationBone of the character

        public float DatabaseDeltaTime { get; private set; }

        private void Update()
        {
            DatabaseDeltaTime = SimulationBone.DatabaseFrameTime;
            // Update the character
            OnUpdate();
            // Update other components depending on the character controller
            if (OnUpdated != null) OnUpdated.Invoke(Time.deltaTime);
        }

        protected void NotifyInputChangedQuickly()
        {
            if (OnInputChangedQuickly != null) OnInputChangedQuickly.Invoke();
        }

        protected abstract void OnUpdate();

        public abstract float3 GetWorldInitPosition();
        public abstract float3 GetWorldInitDirection();
        /// <summary>
        /// Get the prediction in world space of the feature.
        /// e.g. the feature is the position of the character, and it has frames = { 20, 40, 60}
        /// if index=1 it will return the position of the character at frame 40
        /// </summary>
        public abstract float3 GetWorldSpacePrediction(TrajectoryFeature feature, int index);
    }
}