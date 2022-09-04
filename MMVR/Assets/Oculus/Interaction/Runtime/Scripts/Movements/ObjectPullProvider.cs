/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;

namespace Oculus.Interaction.HandPosing
{
    public class ObjectPullProvider : MonoBehaviour, IMovementProvider
    {
        [SerializeField]
        [Min(0f)]
        private float _speed = 1f;
        public float Speed
        {
            get
            {
                return _speed;
            }
            set
            {
                _speed = value;
            }
        }

        [SerializeField]
        [Min(0f)]
        private float _deadZone = 0.02f;
        public float DeadZone
        {
            get
            {
                return _deadZone;
            }
            set
            {
                _deadZone = value;
            }
        }

        public IMovement CreateMovement()
        {
            return new ObjectPull(_speed, _deadZone);
        }
    }

    public class ObjectPull : IMovement
    {
        private float _speed = 1f;
        private float _deadZone = 0f;

        public Pose Pose => _current;
        public bool Stopped => _reachedGrabber;

        private Pose _current = Pose.identity;
        private Pose _grabberStartPose;
        private Pose _grabbableStartPose;
        private Pose _target;

        private Plane _pullingPlane;

        private Vector3 _translationDelta = Vector3.zero;

        private float _lastTime;
        private float _originalDistance;

        private bool _reachedGrabber = false;

        public ObjectPull(float speed, float deadZone)
        {
            _speed = speed;
            _deadZone = deadZone;
        }

        public void MoveTo(Pose target)
        {
            _target = _grabberStartPose = target;
            _current = _grabbableStartPose;
            _lastTime = Time.realtimeSinceStartup;
            _reachedGrabber = false;
            Vector3 grabDir = (_grabbableStartPose.position - _grabberStartPose.position);
            _originalDistance = grabDir.magnitude;
            _pullingPlane = new Plane(grabDir.normalized, _grabberStartPose.position);
        }

        public void UpdateTarget(Pose target)
        {
            _target = target;
        }

        public void StopAndSetPose(Pose source)
        {
            _grabbableStartPose = source;
        }

        public void Tick()
        {
            if (_reachedGrabber)
            {
                _current = _target;
                return;
            }

            float timeDelta = (Time.realtimeSinceStartup - _lastTime);
            _lastTime = Time.realtimeSinceStartup;
            float posDelta = _pullingPlane.GetDistanceToPoint(_target.position);
            if (Mathf.Abs(posDelta) < _deadZone)
            {
                return;
            }

            Vector3 direction = (_current.position - _target.position).normalized;
            _translationDelta = direction * posDelta * _speed * timeDelta;
            float remainingDistance = Vector3.Distance(_current.position, _target.position);
            if (remainingDistance < _translationDelta.magnitude)
            {
                _reachedGrabber = true;
                _current = _target;
            }
            else
            {
                _current.position += _translationDelta;
                float currentDistance = Vector3.Distance(_current.position, _target.position);
                float progress = 1f - Mathf.Clamp01(currentDistance / _originalDistance);
                _current.rotation = Quaternion.Slerp(_grabbableStartPose.rotation, _target.rotation, progress);
            }
        }
    }
}
