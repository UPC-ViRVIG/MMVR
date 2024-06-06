using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputSelector : MonoBehaviour
{
    public GameObject SimulatorRig;
    public GameObject OculusRig;
    public GameObject[] LeftRightCenterBodySimulator;
    public GameObject[] LeftRightCenterBodyOculus;
    public GameObject[] TipLeftRightOculus;
    public GameObject LookAtOculus;
    public GameObject LookAtSimulator;
    public VRCharacterController VRCharacterController;
    public Transform LeftIKTarget;
    public Transform RightIKTarget;
    public Transform LookAtIKTarget;
    public Calibrator Calibrator;
    public Transform OculusCenterEye;
    public Transform SimulatorCenterEye;
    [HideInInspector] public bool IsSimulator;

    private void Update()
    {
        if (IsSimulator)
        {
            LeftIKTarget.position = LeftRightCenterBodySimulator[0].transform.position;
            LeftIKTarget.rotation = LeftRightCenterBodySimulator[0].transform.rotation;
            RightIKTarget.position = LeftRightCenterBodySimulator[1].transform.position;
            RightIKTarget.rotation = LeftRightCenterBodySimulator[1].transform.rotation;
            LookAtIKTarget.position = LookAtSimulator.transform.position;
        }
        else
        {
            LeftIKTarget.position = TipLeftRightOculus[0].transform.position;
            LeftIKTarget.rotation = TipLeftRightOculus[0].transform.rotation;
            RightIKTarget.position = TipLeftRightOculus[1].transform.position;
            RightIKTarget.rotation = TipLeftRightOculus[1].transform.rotation;
            LookAtIKTarget.position = LookAtOculus.transform.position;
        }
    }

    public void SetOculusInput()
    {
        SimulatorRig.SetActive(false);
        OculusRig.SetActive(true);
        VRCharacterController.HMDDevice = LeftRightCenterBodyOculus[2].transform;
        Calibrator.HMD = OculusCenterEye;
        IsSimulator = false;
    }

    public void SetSimulatorInput()
    {
        SimulatorRig.SetActive(true);
        OculusRig.SetActive(false);
        VRCharacterController.HMDDevice = LeftRightCenterBodySimulator[2].transform;
        Calibrator.HMD = SimulatorCenterEye;
        IsSimulator = true;
    }
}

#if UNITY_EDITOR
[UnityEditor.CustomEditor(typeof(InputSelector))]
public class InputSelectorEditor : UnityEditor.Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        
        InputSelector inputSelector = (InputSelector)target;
        
        GUI.enabled = inputSelector.IsSimulator;
        if (GUILayout.Button("Set Oculus Input"))
        {
            inputSelector.SetOculusInput();
            UnityEditor.EditorUtility.SetDirty(inputSelector.SimulatorRig);
            UnityEditor.EditorUtility.SetDirty(inputSelector.OculusRig);
            UnityEditor.EditorUtility.SetDirty(inputSelector.Calibrator.gameObject);
            UnityEditor.EditorUtility.SetDirty(target);
        }
        GUI.enabled = !inputSelector.IsSimulator;
        if (GUILayout.Button("Set Simulator Input"))
        {
            inputSelector.SetSimulatorInput();
            UnityEditor.EditorUtility.SetDirty(inputSelector.SimulatorRig);
            UnityEditor.EditorUtility.SetDirty(inputSelector.OculusRig);
            UnityEditor.EditorUtility.SetDirty(inputSelector.Calibrator.gameObject);
            UnityEditor.EditorUtility.SetDirty(target);
        }
        GUI.enabled = true;
    }
}
#endif