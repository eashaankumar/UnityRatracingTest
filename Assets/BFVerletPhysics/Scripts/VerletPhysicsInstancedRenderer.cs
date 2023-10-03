using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
using Unity.Mathematics;
using Unity.Collections;
using BarelyFunctional.Structs;
using BarelyFunctional.Interfaces;
using BarelyFunctional.VerletPhysics;

// https://github.com/INedelcu/RayTracingMeshInstancingSimple
namespace BarelyFunctional.Renderer
{
    public class VerletPhysicsInstancedRenderer : MonoBehaviour
    {
        [SerializeField]
        BFVerletPhysicsMaster physicsMaster;
        [SerializeField, Range(0.01f, 1f)]
        float resolution = 0.5f;
        [SerializeField]
        TMPro.TMP_Text text;

        public RayTracingShader rayTracingShader = null;

        //public Cubemap envTexture = null;

        [Header("Skybox")]
        [SerializeField]
        UnityEngine.Color topColor;
        [SerializeField]
        UnityEngine.Color bottomColor;

        [Range(1, 100)]
        public uint bounceCountOpaque = 5;

        [Range(1, 100)]
        public uint bounceCountTransparent = 8;

        public Mesh mesh;
        public Material material;

        public Transform target;

        private uint cameraWidth = 0;
        private uint cameraHeight = 0;

        private int convergenceStep = 0;

        private Matrix4x4 prevCameraMatrix;
        private uint prevBounceCountOpaque = 0;
        private uint prevBounceCountTransparent = 0;

        private RenderTexture noisyRadianceRT = null, convergedRT = null;
        private RenderTexture normalRT = null, depthRT = null, albedoRT = null, emissionRT = null,
            specularRT = null, shapeRT = null;

        private RayTracingAccelerationStructure rayTracingAccelerationStructure = null;

        private void CreateRayTracingAccelerationStructure()
        {
            if (rayTracingAccelerationStructure == null)
            {
                RayTracingAccelerationStructure.Settings settings = new RayTracingAccelerationStructure.Settings();
                settings.rayTracingModeMask = RayTracingAccelerationStructure.RayTracingModeMask.Everything;
                settings.managementMode = RayTracingAccelerationStructure.ManagementMode.Automatic;
                settings.layerMask = 255;

                rayTracingAccelerationStructure = new RayTracingAccelerationStructure(settings);
            }
        }

        private void ReleaseResources()
        {
            if (rayTracingAccelerationStructure != null)
            {
                rayTracingAccelerationStructure.Release();
                rayTracingAccelerationStructure = null;
            }

            if (noisyRadianceRT != null)
            {
                ReleaseRT(ref noisyRadianceRT);
                ReleaseRT(ref convergedRT);
                ReleaseRT(ref normalRT);
                ReleaseRT(ref depthRT);
                ReleaseRT(ref albedoRT);
                ReleaseRT(ref emissionRT);
                ReleaseRT(ref specularRT);
                ReleaseRT(ref shapeRT);
            }

            cameraWidth = 0;
            cameraHeight = 0;
        }

        void ReleaseRT(ref RenderTexture tex)
        {
            tex.Release();
            tex = null;
        }

        public int PixelWidth
        {
            get
            {
                //return (int)(Camera.main.pixelWidth * resolution);
                return 426;
            }
        }

        public int PixelHeight
        {
            get
            {
                //return (int)(Camera.main.pixelHeight * resolution);
                return 240;
            }
        }

        private void CreateResources()
        {
            CreateRayTracingAccelerationStructure();

            if (cameraWidth != PixelWidth || cameraHeight != PixelHeight)
            {
                if (noisyRadianceRT)
                {
                    ReleaseRT(ref noisyRadianceRT);
                    ReleaseRT(ref convergedRT);
                    ReleaseRT(ref normalRT);
                    ReleaseRT(ref depthRT);
                    ReleaseRT(ref albedoRT);
                    ReleaseRT(ref emissionRT);
                    ReleaseRT(ref specularRT);
                    ReleaseRT(ref shapeRT);
                }

                RenderTextureDescriptor rtDesc = new RenderTextureDescriptor()
                {
                    dimension = TextureDimension.Tex2D,
                    width = PixelWidth,
                    height = PixelHeight,
                    depthBufferBits = 0,
                    volumeDepth = 1,
                    msaaSamples = 1,
                    vrUsage = VRTextureUsage.OneEye,
                    graphicsFormat = GraphicsFormat.R32G32B32A32_SFloat,
                    enableRandomWrite = true,
                };

                CreateRenderTexture(ref noisyRadianceRT, rtDesc);

                CreateRenderTexture(ref convergedRT, rtDesc);

                CreateRenderTexture(ref normalRT, rtDesc);

                CreateRenderTexture(ref albedoRT, rtDesc);

                CreateRenderTexture(ref depthRT, rtDesc);

                CreateRenderTexture(ref emissionRT, rtDesc);

                CreateRenderTexture(ref specularRT, rtDesc);

                CreateRenderTexture(ref shapeRT, rtDesc);

                cameraWidth = (uint)PixelWidth;
                cameraHeight = (uint)PixelHeight;

                convergenceStep = 0;
            }
        }

        void CreateRenderTexture(ref RenderTexture tex, RenderTextureDescriptor desc)
        {
            tex = new RenderTexture(desc);
            tex.Create();
        }


        void OnDestroy()
        {
            ReleaseResources();
        }

        void OnDisable()
        {
            ReleaseResources();
        }

        private void OnEnable()
        {
            prevCameraMatrix = Camera.main.cameraToWorldMatrix;
            prevBounceCountOpaque = bounceCountOpaque;
            prevBounceCountTransparent = bounceCountTransparent;

        }

        private void Update()
        {
            CreateResources();

            if (Input.GetKeyDown("space"))
                convergenceStep = 0;
        }

       
        [ImageEffectOpaque]
        void OnRenderImage(RenderTexture src, RenderTexture dest)
        {
            if (!SystemInfo.supportsRayTracing || !rayTracingShader)
            {
                Debug.Log("The RayTracing API is not supported by this GPU or by the current graphics API.");
                Graphics.Blit(src, dest);
                return; 
            }

            if (rayTracingAccelerationStructure == null)
                return;

            if (prevCameraMatrix != Camera.main.cameraToWorldMatrix)
                convergenceStep = 0;

            if (prevBounceCountOpaque != bounceCountOpaque)
                convergenceStep = 0;

            if (prevBounceCountTransparent != bounceCountTransparent)
                convergenceStep = 0;

            convergenceStep = 3;

            rayTracingAccelerationStructure.ClearInstances();

            #region Instancing

            VoxelInstancedRenderer vRenderer = physicsMaster.Tick();
            /*vRenderer.data[0] = new Data { color = new float3(1, 0, 0), emission = Mathf.Sin(Time.time) * 0.5f + 0.5f };
            vRenderer.data[1] = new Data { color = new float3(1, 1, 0), emission = Mathf.Cos(Time.time) * 0.5f + 0.5f };
            vRenderer.matrices[0] = float4x4.TRS(target.position, target.rotation, target.localScale);
            vRenderer.matrices[1] = float4x4.TRS(target.position + target.forward * 2, target.rotation, target.localScale);*/
            GraphicsBuffer data = null;
            if (vRenderer.standardMaterialData.Length > 0)
            {
                data = new GraphicsBuffer(GraphicsBuffer.Target.Structured, vRenderer.standardMaterialData.Length, StandardMaterialData.Size);
                data.SetData(vRenderer.standardMaterialData);

                RayTracingMeshInstanceConfig config = new RayTracingMeshInstanceConfig(mesh, 0, material);

                config.materialProperties = new MaterialPropertyBlock();
                config.materialProperties.SetBuffer("g_Data", data);
                config.material.enableInstancing = true;

                rayTracingAccelerationStructure.AddInstances(config, vRenderer.standardMatrices);

            }
            text.text = $"Voxels {vRenderer.standardMaterialData.Length}\n{SystemInfo.graphicsDeviceName}\n{SystemInfo.graphicsDeviceType}\n{noisyRadianceRT.width}x{noisyRadianceRT.height}";

            vRenderer.Dispose();
            #endregion

            // Not really needed per frame if the scene is static.
            rayTracingAccelerationStructure.Build();

            rayTracingShader.SetShaderPass("PathTracing");

            Shader.SetGlobalInt(Shader.PropertyToID("g_BounceCountOpaque"), (int)bounceCountOpaque);
            Shader.SetGlobalInt(Shader.PropertyToID("g_BounceCountTransparent"), (int)bounceCountTransparent);

            // Input
            rayTracingShader.SetAccelerationStructure(Shader.PropertyToID("g_AccelStruct"), rayTracingAccelerationStructure);
            rayTracingShader.SetFloat(Shader.PropertyToID("g_Zoom"), Mathf.Tan(Mathf.Deg2Rad * Camera.main.fieldOfView * 0.5f));
            rayTracingShader.SetFloat(Shader.PropertyToID("g_AspectRatio"), cameraWidth / (float)cameraHeight);
            rayTracingShader.SetInt(Shader.PropertyToID("g_ConvergenceStep"), convergenceStep);
            rayTracingShader.SetInt(Shader.PropertyToID("g_FrameIndex"), Time.frameCount);
            rayTracingShader.SetVector(Shader.PropertyToID("g_SkyboxBottomColor"), new Vector3(bottomColor.r, bottomColor.g, bottomColor.b));
            rayTracingShader.SetVector(Shader.PropertyToID("g_SkyboxTopColor"), new Vector3(topColor.r, topColor.g, topColor.b));

            //rayTracingShader.SetTexture(Shader.PropertyToID("g_EnvTex"), envTexture);

            // Output
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Radiance"), noisyRadianceRT);
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Normal"), normalRT);
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Albedo"), albedoRT);
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Depth"), depthRT);
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Emission"), emissionRT);
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Specular"), specularRT);
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Shape"), shapeRT);

            // Output
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Radiance"), noisyRadianceRT);

            for (int i = 0; i < 100; i++)
            {

                rayTracingShader.SetInt(Shader.PropertyToID("g_ConvergenceStep"), convergenceStep);

                rayTracingShader.Dispatch("MainRayGenShader", (int)cameraWidth, (int)cameraHeight, 1, Camera.main);

                convergenceStep++;

            }

            Graphics.Blit(noisyRadianceRT, dest);


            prevCameraMatrix = Camera.main.cameraToWorldMatrix;
            prevBounceCountOpaque = bounceCountOpaque;
            prevBounceCountTransparent = bounceCountTransparent;
            if (data != null) data.Release();
        }
    }
}
