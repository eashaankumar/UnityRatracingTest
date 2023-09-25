using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
using Unity.Mathematics;
using Unity.Collections;
using BarelyFunctional.Structs;
using BarelyFunctional.Interfaces;
using BarelyFunctional.VerletPhysics;

// https://github.com/INedelcu/RayTracingMeshInstancingSimple
namespace BarelyFunctional.Renderer.Denoiser.DataGeneration
{
    public class DatasetRenderer : MonoBehaviour
    {
        [SerializeField]
        BFVerletPhysicsMaster physicsMaster;
        [SerializeField]
        Dataset trainDataset;
        [SerializeField]
        UnityEngine.Color topColor;
        [SerializeField]
        UnityEngine.Color bottomColor;

        public RayTracingShader rayTracingShader = null;

        //public Cubemap envTexture = null;

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

        private RenderTexture rayTracingOutput = null;

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

            if (rayTracingOutput != null)
            {
                rayTracingOutput.Release();
                rayTracingOutput = null;
            }

            cameraWidth = 0;
            cameraHeight = 0;
        }

        public int PixelWidth
        {
            get
            {
                return trainDataset.PixelWidth;
            }
        }

        public int PixelHeight
        {
            get
            {
                return trainDataset.PixelHeight;
            }
        }

        private void CreateResources()
        {
            CreateRayTracingAccelerationStructure();

            if (cameraWidth != PixelWidth || cameraHeight != PixelHeight)
            {
                if (rayTracingOutput)
                    rayTracingOutput.Release();

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

                rayTracingOutput = new RenderTexture(rtDesc);
                rayTracingOutput.Create();

                cameraWidth = (uint)PixelWidth;
                cameraHeight = (uint)PixelHeight;

                //convergenceStep = 0;
            }
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

       
        [ImageEffectOpaque]
        void OnRenderImage(RenderTexture src, RenderTexture dest)
        {
            if (!SystemInfo.supportsRayTracing || !rayTracingShader)
            {
                Debug.Log("The RayTracing API is not supported by this GPU or by the current graphics API.");
                Graphics.Blit(src, dest);
                return; 
            }

            // TODO: Handle convergence
            
            CreateResources();

            if (rayTracingAccelerationStructure == null)
                return;

            if (prevCameraMatrix != Camera.main.cameraToWorldMatrix)
                convergenceStep = 0;

            if (prevBounceCountOpaque != bounceCountOpaque)
                convergenceStep = 0;

            if (prevBounceCountTransparent != bounceCountTransparent)
                convergenceStep = 0;

            convergenceStep = 0;

            rayTracingAccelerationStructure.ClearInstances();

            #region Instancing

            VerletPhysicsRenderer vRenderer = physicsMaster.Tick();
            GraphicsBuffer data = null;
            if (vRenderer.data.Length > 0)
            {
                data = new GraphicsBuffer(GraphicsBuffer.Target.Structured, vRenderer.data.Length, 4 * sizeof(float));
                data.SetData(vRenderer.data);

                RayTracingMeshInstanceConfig config = new RayTracingMeshInstanceConfig(mesh, 0, material);

                config.materialProperties = new MaterialPropertyBlock();
                config.materialProperties.SetBuffer("g_Data", data);
                config.material.enableInstancing = true;

                rayTracingAccelerationStructure.AddInstances(config, vRenderer.matrices);

            }
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
            rayTracingShader.SetTexture(Shader.PropertyToID("g_Radiance"), rayTracingOutput);

            //for (int i = 0; i < 10; i++)
            {

                rayTracingShader.SetInt(Shader.PropertyToID("g_ConvergenceStep"), convergenceStep);

                rayTracingShader.Dispatch("MainRayGenShader", (int)cameraWidth, (int)cameraHeight, 1, Camera.main);

                convergenceStep++;
                 
            }

            Graphics.Blit(rayTracingOutput, dest);


            prevCameraMatrix = Camera.main.cameraToWorldMatrix;
            prevBounceCountOpaque = bounceCountOpaque;
            prevBounceCountTransparent = bounceCountTransparent;
            if (data != null) data.Release();
        }
    }
}
