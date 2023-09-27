using BarelyFunctional.Structs;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class WorldGenerator : MonoBehaviour
{
    [SerializeField]
    uint seed;
    [SerializeField]
    float2 spawnRadiusMinMax;
    [SerializeField]
    float2 spawnSizeMinMax;
    [SerializeField]
    uint numVoxels;

    VoxelInstancedRenderer rendererCache;
    bool isReady;

    public bool IsReady { get { return isReady; } }
    public VoxelInstancedRenderer RendererCache { get { return rendererCache; } }
    
    IEnumerator Start()
    {
        isReady = false;
        VerletPhysicsRendererAssembler assembler = new VerletPhysicsRendererAssembler((int)numVoxels, (int)numVoxels, Allocator.TempJob);
        GenerateWorldJob generateWorldJob = new GenerateWorldJob
        {
            standardMaterialAssembly = assembler.standardMaterialAssembly.AsParallelWriter(),
            glassMaterialAssembly = assembler.glassMaterialAssembly.AsParallelWriter(),
            random = new Unity.Mathematics.Random(seed),
            spawnRadius = spawnRadiusMinMax,
            spawnSize = spawnSizeMinMax
        };
        JobHandle genHandle = generateWorldJob.Schedule(assembler.standardMaterialAssembly.Length, 64);
        yield return new WaitUntil(() => genHandle.IsCompleted);
        genHandle.Complete();

        if (rendererCache.IsCreated) rendererCache.Dispose();
        rendererCache = new VoxelInstancedRenderer(assembler.standardMaterialAssembly.Length, assembler.glassMaterialAssembly.Length, Allocator.Persistent);
        PopulateRendererCacheJob rendererCacheJob = new PopulateRendererCacheJob
        {
            assembler = assembler,
            rendererCache = rendererCache
        };
        JobHandle cacheHandle = rendererCacheJob.Schedule(assembler.standardMaterialAssembly.Length + assembler.glassMaterialAssembly.Length, 64);
        yield return new WaitUntil(() => cacheHandle.IsCompleted);
        cacheHandle.Complete();

        assembler.Dispose();
        isReady = true;
    }

    private void OnDestroy()
    {
        if (RendererCache.IsCreated) RendererCache.Dispose();
    }

    [BurstCompile]
    struct GenerateWorldJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction]
        public NativeList<StandardVoxelAssembledData>.ParallelWriter standardMaterialAssembly;
        [NativeDisableParallelForRestriction]
        public NativeList<GlassVoxelAssembledData>.ParallelWriter glassMaterialAssembly;
        public Unity.Mathematics.Random random;
        [ReadOnly]
        public float2 spawnRadius;
        [ReadOnly]
        public float2 spawnSize;

        public void Execute(int index)
        {
            float3 pos = random.NextFloat3Direction() * random.NextFloat(spawnRadius.x, spawnRadius.y);
            quaternion rot = random.NextQuaternionRotation();
            float size = random.NextFloat(spawnSize.x, spawnSize.y);
            Matrix4x4 trs = Matrix4x4.TRS(pos, rot, new float3(1, 1, 1) * size);

            VoxelMaterialType type = (VoxelMaterialType)random.NextInt(0, 2);

            if (type == VoxelMaterialType.STANDARD)
            {
                standardMaterialAssembly.AddNoResize(
                    new StandardVoxelAssembledData
                    {
                        material = new StandardMaterialData
                        {
                            albedo = RandColor(),
                            specular = RandSpecular(),
                            emission = RandColor() * random.NextFloat(0f, 1f),
                            smoothness = 0.5f,
                            metallic = 0.2f,
                            ior = 0.6f
                        },
                        trs = trs
                    }
                );
            }
            else
            {
                glassMaterialAssembly.AddNoResize(
                    new GlassVoxelAssembledData
                    {
                        material= new GlassMaterialData
                        {
                            albedo = RandColor(),
                            emission = RandColor() * random.NextFloat(0f, 1.1f),
                            ior = random.NextFloat(1.0f, 2.8f),
                            roughness = random.NextFloat(0f, 0.5f),
                            extinctionCoeff = 0,
                            flatShading = 0,
                        },
                        trs= trs
                    }
                );
            }
        }

        float3 RandColor()
        {
            return new float3(math.sin(Time.time) * 0.5f + 0.5f, math.cos(Time.time) * 0.5f + 0.5f, math.tan(Time.time) * 0.5f + 0.5f);
        }

        float3 RandSpecular()
        {
            return new float3(0.5f, 0.5f, 0.5f);
        }
    }

    [BurstCompile]
    struct PopulateRendererCacheJob : IJobParallelFor
    {
        [ReadOnly] public VerletPhysicsRendererAssembler assembler;
        [NativeDisableParallelForRestriction] public VoxelInstancedRenderer rendererCache;
        public void Execute(int index)
        {
            if (index < assembler.standardMaterialAssembly.Length)
            {
                StandardVoxelAssembledData assembly = assembler.standardMaterialAssembly[index];
                rendererCache.standardMatrices[index] = assembly.trs;
                rendererCache.standardMaterialData[index] = assembly.material;
            }
            else if (index < assembler.standardMaterialAssembly.Length + assembler.glassMaterialAssembly.Length)
            {
                int idx = index - assembler.standardMaterialAssembly.Length;
                if (idx < 0 || idx >= assembler.glassMaterialAssembly.Length) return;
                GlassVoxelAssembledData assembly = assembler.glassMaterialAssembly[idx];
                rendererCache.glassMatrices[idx] = assembly.trs;
                rendererCache.glassMaterialData[idx] = assembly.material;
            }
        }
    }
}
