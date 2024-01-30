using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Calibrator : MonoBehaviour
{
    public Transform HMD;
    public Transform Avatar;
    public Transform MotionMatching;
    public Transform AvatarEyes;
    public float EyesHeight { get; private set; }
    public float HMDHeight
    {
        get
        {
            return HMD.position.y;
        }
    }

    private static readonly float CALIBRATE_COOLDOWN = 2.0f;
    private float LastCalibrateTime = -CALIBRATE_COOLDOWN;

    private void Awake()
    {
        EyesHeight = 1.8f; // Default Value
    }

    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch) && LastCalibrateTime + CALIBRATE_COOLDOWN < Time.time)
        {
            Calibrate();
        }
    }

    public void Calibrate()
    {
        EyesHeight = HMDHeight;
        Vector3 localScale = new Vector3(1, 1, 1) * (EyesHeight / AvatarEyes.position.y);
        Avatar.localScale = localScale;
        MotionMatching.localScale = localScale;
        LastCalibrateTime = Time.time;
    }
}

#if UNITY_EDITOR

[UnityEditor.CustomEditor(typeof(Calibrator))]
public class CalibratorEditor : UnityEditor.Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        if (!Application.isPlaying)
        {
            return;
        }

        Calibrator calibrator = (Calibrator)target;
        if (GUILayout.Button("Calibrate"))
        {
            calibrator.Calibrate();
        }
    }
}

#endif
