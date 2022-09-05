using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Calibrator : MonoBehaviour
{
    public Transform HMD;
    public Transform Avatar;
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
            EyesHeight = HMDHeight;
            Avatar.localScale = new Vector3(1, 1, 1) * (EyesHeight / AvatarEyes.position.y);
            LastCalibrateTime = Time.time;
        }
    }
}
