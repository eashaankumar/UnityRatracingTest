using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
using Unity.Mathematics;
using Unity.Collections;
using BarelyFunctional.Structs;
using BarelyFunctional.Interfaces;
using BarelyFunctional.VerletPhysics;
using UnityEngine.UI;
using TMPro;
using System.Collections;

// https://github.com/INedelcu/RayTracingMeshInstancingSimple
namespace BarelyFunctional.Renderer.Denoiser.DataGeneration
{
    public class DatasetRenderer : MonoBehaviour
    {
        [Header("Renderer Cache Generator")]
        [SerializeField]
        WorldGenerator worldGenerator;

        [SerializeField]
        Dataset trainDataset;

        [Header("Skybox")]
        [SerializeField]
        UnityEngine.Color topColor;
        [SerializeField]
        UnityEngine.Color bottomColor;

        [Header("UI")]
        [SerializeField]
        RawImages images;
        [SerializeField]
        Canvas iamgesCanvas;
        [SerializeField]
        TMP_Text convergence;

        [System.Serializable]
        struct RawImages
        {
            [SerializeField]
            public RawImage displayImage;
            [SerializeField]
            public RawImage normalImage;
            [SerializeField]
            public RawImage depthImage;
            [SerializeField]
            public RawImage albedoImage;
            [SerializeField]
            public RawImage emissionImage;
            [SerializeField]
            public RawImage materialImage;
            [SerializeField]
            public RawImage noisyImage;
            [SerializeField]
            public RawImage convergedImage;
        }

        [System.Serializable]
        public enum ImageType
        {
            NOISY, CONVERGED, NORMAL, ALBEDO, DEPTH, EMISSION, MATERIAL
        }

        [Header("Ray Tracing")]
        public RayTracingShader rayTracingShader = null;

        //public Cubemap envTexture = null;

        [Range(1, 100)]
        public uint bounceCountOpaque = 5;

        [Range(1, 100)]
        public uint bounceCountTransparent = 8;

        public Mesh mesh;
        public Material material;
        public Material glassMaterial;

        private uint cameraWidth = 0;
        private uint cameraHeight = 0;

        private Matrix4x4 prevCameraMatrix;
        private uint prevBounceCountOpaque = 0;
        private uint prevBounceCountTransparent = 0;

        private RenderTexture noisyRadianceRT = null, convergedRT = null;
        private RenderTexture normalRT = null, depthRT = null, albedoRT = null, emissionRT = null,
            specularRT = null;

        private RayTracingAccelerationStructure rayTracingAccelerationStructure = null;

        ImageType imageType = ImageType.NOISY;

        VoxelInstancedRenderer vRenderer;
        GraphicsBuffer stadardMaterialdata = null, glassMaterialData = null;

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
                if (noisyRadianceRT)
                {
                    //noisyRadianceRT.Release();
                    ReleaseRT(ref noisyRadianceRT);
                    ReleaseRT(ref convergedRT);
                    ReleaseRT(ref normalRT);
                    ReleaseRT(ref depthRT);
                    ReleaseRT(ref albedoRT);
                    ReleaseRT(ref emissionRT);
                    ReleaseRT(ref specularRT);
                }

                RenderTextureDescriptor rtDesc4Channel = new RenderTextureDescriptor()
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

                CreateRenderTexture(ref noisyRadianceRT, rtDesc4Channel);

                CreateRenderTexture(ref convergedRT, rtDesc4Channel);

                CreateRenderTexture(ref normalRT, rtDesc4Channel);

                CreateRenderTexture(ref albedoRT, rtDesc4Channel);

                CreateRenderTexture(ref depthRT, rtDesc4Channel);

                CreateRenderTexture(ref emissionRT, rtDesc4Channel);

                CreateRenderTexture(ref specularRT, rtDesc4Channel);

                cameraWidth = (uint)PixelWidth;
                cameraHeight = (uint)PixelHeight; 

                //convergenceStep = 0;
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
            vRenderer.Dispose();
            if (stadardMaterialdata != null) stadardMaterialdata.Release();
            if (glassMaterialData != null) glassMaterialData.Release();
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

        
        IEnumerator Start()
        {
            yield return new WaitUntil(() => worldGenerator.IsReady);
            vRenderer = worldGenerator.RendererCache;

            CreateResources();

            rayTracingAccelerationStructure.ClearInstances();
            
            if (vRenderer.standardMaterialData.Length > 0)
            {
                stadardMaterialdata = new GraphicsBuffer(GraphicsBuffer.Target.Structured, vRenderer.standardMaterialData.Length, StandardMaterialData.Size);
                stadardMaterialdata.SetData(vRenderer.standardMaterialData);

                RayTracingMeshInstanceConfig config = new RayTracingMeshInstanceConfig(mesh, 0, material);
                config.materialProperties = new MaterialPropertyBlock();
                config.materialProperties.SetBuffer("g_Data", stadardMaterialdata);
                config.material.enableInstancing = true;

                rayTracingAccelerationStructure.AddInstances(config, vRenderer.standardMatrices);

            }

            if (vRenderer.glassMaterialData.Length > 0)
            {
                glassMaterialData = new GraphicsBuffer(GraphicsBuffer.Target.Structured, vRenderer.glassMaterialData.Length, GlassMaterialData.Size);
                glassMaterialData.SetData(vRenderer.glassMaterialData);

                RayTracingMeshInstanceConfig config = new RayTracingMeshInstanceConfig(mesh, 0, glassMaterial);
                config.materialProperties = new MaterialPropertyBlock();
                config.materialProperties.SetBuffer("g_Data", glassMaterialData);
                config.material.enableInstancing = true;

                rayTracingAccelerationStructure.AddInstances(config, vRenderer.glassMatrices);
            }

            // Not really needed per frame if the scene is static.
            rayTracingAccelerationStructure.Build();
            StartCoroutine(GenDataSet());
        }
        

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.M))
            {
                iamgesCanvas.enabled = !iamgesCanvas.enabled;
            }
        }


        IEnumerator GenDataSet()
        {
            if (!SystemInfo.supportsRayTracing || !rayTracingShader)
            {
                Debug.Log("The RayTracing API is not supported by this GPU or by the current graphics API.");
                //Graphics.Blit(src, dest);
                yield break;
            }
            while (rayTracingAccelerationStructure == null)
                yield return null;
            int numSamples = 1000;
            Camera main = Camera.main;
            for (int sample = 0; sample < numSamples; sample++)
            {
                int convergenceStep = 0;

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

                rayTracingShader.SetInt(Shader.PropertyToID("g_ConvergenceStep"), convergenceStep);

                rayTracingShader.Dispatch("MainRayGenShader", (int)cameraWidth, (int)cameraHeight, 1, Camera.main);

                convergenceStep++;

                for (int i = 0; i < trainDataset.Convergence; i++)
                {
                    rayTracingShader.SetTexture(Shader.PropertyToID("g_Radiance"), convergedRT);

                    rayTracingShader.SetInt(Shader.PropertyToID("g_ConvergenceStep"), convergenceStep);

                    rayTracingShader.Dispatch("MainRayGenShader", (int)cameraWidth, (int)cameraHeight, 1, Camera.main);
                    convergenceStep++;
                    yield return null;


                    convergence.text = $"Convergence: {convergenceStep}";

                    switch (imageType)
                    {
                        case ImageType.NOISY:
                            Graphics.Blit(noisyRadianceRT, main.targetTexture);
                            images.displayImage.texture = noisyRadianceRT;
                            break;
                        case ImageType.CONVERGED:
                            Graphics.Blit(convergedRT, main.targetTexture);
                            images.displayImage.texture = convergedRT;
                            break;
                        case ImageType.NORMAL:
                            Graphics.Blit(normalRT, main.targetTexture);
                            images.displayImage.texture = normalRT;
                            break;
                        case ImageType.ALBEDO:
                            Graphics.Blit(albedoRT, main.targetTexture);
                            images.displayImage.texture = albedoRT;
                            break;
                        case ImageType.DEPTH:
                            Graphics.Blit(depthRT, main.targetTexture);
                            images.displayImage.texture = depthRT;
                            break;
                        case ImageType.EMISSION:
                            Graphics.Blit(emissionRT, main.targetTexture);
                            images.displayImage.texture = emissionRT;
                            break;
                        case ImageType.MATERIAL:
                            Graphics.Blit(specularRT, main.targetTexture);
                            images.displayImage.texture = specularRT;
                            break;
                    }

                    images.normalImage.texture = normalRT;
                    images.albedoImage.texture = albedoRT;
                    images.depthImage.texture = depthRT;
                    images.emissionImage.texture = emissionRT;
                    images.materialImage.texture = specularRT;
                    images.noisyImage.texture = noisyRadianceRT;
                    images.convergedImage.texture = convergedRT;
                }
                // invoke dataset

            }
        }


        public void SetTexture(int imageType)
        {
            this.imageType = (ImageType) imageType;
        }
    }
}
