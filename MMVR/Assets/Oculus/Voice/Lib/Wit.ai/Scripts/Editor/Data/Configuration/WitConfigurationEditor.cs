﻿/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using Facebook.WitAi.Configuration;
using Facebook.WitAi.Data.Configuration;
using UnityEditor;
using UnityEngine;

namespace Facebook.WitAi.Windows
{
    public class WitConfigurationEditor : Editor
    {
        public WitConfiguration configuration { get; private set; }
        private string serverToken;
        private bool initialized = false;
        public bool drawHeader = true;
        private bool foldout = true;
        private int requestTab = -1;

        // Tab IDs
        protected const string TAB_APPLICATION_ID = "application";
        protected const string TAB_INTENTS_ID = "intents";
        protected const string TAB_ENTITIES_ID = "entities";
        protected const string TAB_TRAITS_ID = "traits";
        private string[] _tabIds = new string[] { TAB_APPLICATION_ID, TAB_INTENTS_ID, TAB_ENTITIES_ID, TAB_TRAITS_ID };

        public virtual Texture2D HeaderIcon => WitStyles.HeaderIcon;
        public virtual string HeaderUrl => WitStyles.GetAppURL(WitConfigurationUtility.GetAppID(configuration), WitStyles.WitAppEndpointType.Settings);
        public virtual string OpenButtonLabel => WitStyles.Texts.WitOpenButtonLabel;

        public void Initialize()
        {
            // Refresh configuration & auth tokens
            configuration = target as WitConfiguration;
            // Get app server token
            serverToken = WitAuthUtility.GetAppServerToken(configuration);
            if (CanConfigurationRefresh(configuration) && WitConfigurationUtility.IsServerTokenValid(serverToken))
            {
                // Get client token if needed
                string appID = WitConfigurationUtility.GetAppID(configuration);
                if (string.IsNullOrEmpty(appID))
                {
                    configuration.SetServerToken(serverToken);
                }
                // Refresh additional data
                else
                {
                    SafeRefresh();
                }
            }
        }

        public override void OnInspectorGUI()
        {
            // Init styles
            WitStyles.Init();
            // Init if needed
            if (!initialized || configuration != target)
            {
                Initialize();
                initialized = true;
            }

            // Draw header
            if (drawHeader)
            {
                WitEditorUI.LayoutHeaderButton(HeaderIcon, HeaderUrl);
                GUILayout.Space(WitStyles.HeaderPaddingBottom);
                EditorGUI.indentLevel++;
            }

            // Layout content
            LayoutContent();

            // Undent
            if (drawHeader)
            {
                EditorGUI.indentLevel--;
            }
        }

        protected virtual void LayoutContent()
        {
            // Begin vertical box
            GUILayout.BeginVertical(EditorStyles.helpBox);

            // Title Foldout
            GUILayout.BeginHorizontal();
            string foldoutText = WitStyles.Texts.ConfigurationHeaderLabel;
            string appName = configuration?.application?.name;
            if (!string.IsNullOrEmpty(appName))
            {
                foldoutText = foldoutText + " - " + appName;
            }
            foldout = WitEditorUI.LayoutFoldout(new GUIContent(foldoutText), foldout);
            // Refresh button
            if (CanConfigurationRefresh(configuration))
            {
                if (string.IsNullOrEmpty(appName))
                {
                    bool isValid =  WitConfigurationUtility.IsServerTokenValid(serverToken);
                    GUI.enabled = isValid;
                    if (WitEditorUI.LayoutTextButton(WitStyles.Texts.ConfigurationRefreshButtonLabel))
                    {
                        ApplyServerToken(serverToken);
                    }
                }
                else
                {
                    bool isRefreshing = configuration.IsRefreshingData();
                    GUI.enabled = !isRefreshing;
                    if (WitEditorUI.LayoutTextButton(isRefreshing ? WitStyles.Texts.ConfigurationRefreshingButtonLabel : WitStyles.Texts.ConfigurationRefreshButtonLabel))
                    {
                        SafeRefresh();
                    }
                }
            }
            GUI.enabled = true;
            GUILayout.EndHorizontal();
            GUILayout.Space(WitStyles.ButtonMargin);

            // Show configuration app data
            if (foldout)
            {
                // Indent
                EditorGUI.indentLevel++;

                // Server access token
                bool updated = false;
                WitEditorUI.LayoutPasswordField(WitStyles.ConfigurationServerTokenContent, ref serverToken, ref updated);
                if (updated)
                {
                    ApplyServerToken(serverToken);
                }

                // Additional data
                if (configuration)
                {
                    LayoutConfigurationData();
                }

                // Undent
                EditorGUI.indentLevel--;
            }

            // End vertical box layout
            GUILayout.EndVertical();

            // Layout configuration request tabs
            LayoutConfigurationRequestTabs();

            // Additional open wit button
            GUILayout.FlexibleSpace();
            if (GUILayout.Button(OpenButtonLabel, WitStyles.TextButton))
            {
                Application.OpenURL(HeaderUrl);
            }
        }
        // Apply server token
        public void ApplyServerToken(string newToken)
        {
            serverToken = newToken;
            configuration.SetServerToken(serverToken);
        }
        // Whether or not to allow a configuration to refresh
        protected virtual bool CanConfigurationRefresh(WitConfiguration configuration)
        {
            return configuration;
        }
        // Layout configuration data
        protected virtual void LayoutConfigurationData()
        {
            // Reset update
            bool updated = false;
            // Client access field
            WitEditorUI.LayoutPasswordField(WitStyles.ConfigurationClientTokenContent, ref configuration.clientAccessToken, ref updated);
            if (updated && string.IsNullOrEmpty(configuration.clientAccessToken))
            {
                Debug.LogError("Client access token is not defined. Cannot perform requests with '" + configuration.name + "'.");
            }
            // Timeout field
            WitEditorUI.LayoutIntField(WitStyles.ConfigurationRequestTimeoutContent, ref configuration.timeoutMS, ref updated);
            // Updated
            if (updated)
            {
                EditorUtility.SetDirty(configuration);
            }

            // Show configuration app data
            LayoutConfigurationEndpoint();
        }
        // Layout endpoint data
        protected virtual void LayoutConfigurationEndpoint()
        {
            // Generate if needed
            if (configuration.endpointConfiguration == null)
            {
                configuration.endpointConfiguration = new WitEndpointConfig();
                EditorUtility.SetDirty(configuration);
            }

            // Handle via serialized object
            var serializedObj = new SerializedObject(configuration);
            var serializedProp = serializedObj.FindProperty("endpointConfiguration");
            EditorGUILayout.PropertyField(serializedProp);
            serializedObj.ApplyModifiedProperties();
        }
        // Tabs
        protected virtual void LayoutConfigurationRequestTabs()
        {
            // Indent
            EditorGUI.indentLevel++;

            // Iterate tabs
            if (_tabIds != null)
            {
                GUILayout.BeginHorizontal();
                for (int i = 0; i < _tabIds.Length; i++)
                {
                    // Enable if not selected
                    GUI.enabled = requestTab != i;
                    // If valid and clicked, begin selecting
                    string tabPropertyID = _tabIds[i];
                    if (ShouldTabShow(configuration, tabPropertyID))
                    {
                        if (WitEditorUI.LayoutTabButton(GetTabText(configuration, tabPropertyID, true)))
                        {
                            requestTab = i;
                        }
                    }
                    // If invalid, stop selecting
                    else if (requestTab == i)
                    {
                        requestTab = -1;
                    }
                }
                GUI.enabled = true;
                GUILayout.EndHorizontal();
            }

            // Layout selected tab using property id
            string propertyID = requestTab >= 0 && requestTab < _tabIds.Length ? _tabIds[requestTab] : string.Empty;
            if (!string.IsNullOrEmpty(propertyID))
            {
                SerializedObject serializedObj = new SerializedObject(configuration);
                SerializedProperty serializedProp = serializedObj.FindProperty(propertyID);
                if (serializedProp == null)
                {
                    WitEditorUI.LayoutErrorLabel(GetTabText(configuration, propertyID, false));
                }
                else if (!serializedProp.isArray)
                {
                    EditorGUILayout.PropertyField(serializedProp);
                }
                else if (serializedProp.arraySize == 0)
                {
                    WitEditorUI.LayoutErrorLabel(GetTabText(configuration, propertyID, false));
                }
                else
                {
                    for (int i = 0; i < serializedProp.arraySize; i++)
                    {
                        SerializedProperty serializedPropChild = serializedProp.GetArrayElementAtIndex(i);
                        EditorGUILayout.PropertyField(serializedPropChild);
                    }
                }
                serializedObj.ApplyModifiedProperties();
            }

            // Undent
            EditorGUI.indentLevel--;
        }
        // Determine if tab should show
        protected virtual bool ShouldTabShow(WitConfiguration configuration, string tabID)
        {
            return true;
        }
        // Get tab text
        protected virtual string GetTabText(WitConfiguration configuration, string tabID, bool titleLabel)
        {
            switch (tabID)
            {
                case TAB_APPLICATION_ID:
                    return titleLabel ? WitStyles.Texts.ConfigurationApplicationTabLabel : WitStyles.Texts.ConfigurationApplicationMissingLabel;
                case TAB_INTENTS_ID:
                    return titleLabel ? WitStyles.Texts.ConfigurationIntentsTabLabel : WitStyles.Texts.ConfigurationIntentsMissingLabel;
                case TAB_ENTITIES_ID:
                    return titleLabel ? WitStyles.Texts.ConfigurationEntitiesTabLabel : WitStyles.Texts.ConfigurationEntitiesMissingLabel;
                case TAB_TRAITS_ID:
                    return titleLabel ? WitStyles.Texts.ConfigurationTraitsTabLabel : WitStyles.Texts.ConfigurationTraitsMissingLabel;
            }
            return string.Empty;
        }
        // Safe refresh
        protected virtual void SafeRefresh()
        {
            if (WitConfigurationUtility.IsServerTokenValid(serverToken))
            {
                configuration.SetServerToken(serverToken);
            }
            else if (WitConfigurationUtility.IsClientTokenValid(configuration.clientAccessToken))
            {
                configuration.RefreshData();
            }
        }
    }
}
