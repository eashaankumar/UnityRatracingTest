using BarelyFunctional.Interfaces;
using BarelyFunctional.Structs;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace BarelyFunctional.VerletPhysics
{ 
    public class BFVerletPhysicsMaster : MonoBehaviour
    {
        [SerializeField]
        double3 gravity;
        [SerializeField, Min(0.0001f)]
        double simDt;
        [SerializeField, Range(1, 100)]
        int subSteps;
        [SerializeField, Range(0.1f, 10)]
        double cellSize;

        public int count;

        //VerletPhysicsSimulatorJob simulatorJob;
        VerletPhysicsWorld verletWorld;
        VoxelWorld voxelWorld;
        VerletPhysicsToRendererAssemblerJob physicsToRendererAssemblerJob;
        VerletPhysicsHasherJob verletPhysicsHasher;
        VerletPhysicsCollisionsJob verletPhysicsCollisions;
        float lastTick;

        Unity.Mathematics.Random random;

        private void OnDestroy()
        {
            verletWorld.Dispose();
            voxelWorld.Dispose();
        }

        private void Awake()
        {
            verletWorld = new VerletPhysicsWorld(gravity, Allocator.Persistent);
            voxelWorld = new VoxelWorld(12424, Allocator.Persistent);
            //simulatorJob = new VerletPhysicsSimulatorJob();
            verletPhysicsHasher = new VerletPhysicsHasherJob();
            physicsToRendererAssemblerJob = new VerletPhysicsToRendererAssemblerJob();
            verletPhysicsCollisions = new VerletPhysicsCollisionsJob();
            random = new Unity.Mathematics.Random(1224214);

            BuildGround();
        }

        void BuildGround()
        {
            int2 size = new int2(25, 25);
            for(int x = -size.x/2; x <= size.x/2; x++)
            {
                for (int z = -size.y / 2; z <= size.y / 2; z++)
                {
                    float voxelSize = 2.0f;
                    verletWorld.AddParticle(
                    new VerletParticle
                    {
                        drag = 0,
                        mass = 1.0,
                        radius = voxelSize / 2,
                        pos_current = new double3(x, -10, z) * 2,
                        freeze = true,
                    });
                    voxelWorld.AddVoxel(
                        voxelSize,
                        new StandardMaterialData
                        {
                            albedo = 1,
                            specular = 0,
                            emission = (x % 2 == 0 && z % 2 == 0) ? 1 : 0,
                            smoothness = 0.5f,
                            metallic = 0.2f,
                            ior = 0.6f
                        },
                        (uint)(Time.time * 1000)
                        );
                }
            }
        }

        float3 RandColor()
        {
            return new float3(math.sin(Time.time) * 0.5f + 0.5f, math.cos(Time.time) * 0.5f + 0.5f, math.tan(Time.time) * 0.5f + 0.5f);
        }

        float3 RandSpecular()
        {
            return new float3(0.5f,0.5f, 0.5f);
        }

        public VerletPhysicsRenderer Tick()
        {
            float time = Time.time;
            if (time - lastTick >= simDt)
            {
                if (random.NextFloat(0f, 1f) < 0.1f)
                {
                    float voxelSize = random.NextFloat(0.75f, 1.0f);
                    double3 pos = new double3(math.sin(verletWorld.ParticleCount * 2), 0, math.cos(verletWorld.ParticleCount * 2)) * 5;
                    verletWorld.AddParticle(
                        new VerletParticle
                        {
                            drag = 0.1,
                            mass = 1.0,
                            pos_current = pos + new double3(0,0.001,0),
                            pos_old = pos,
                            radius = voxelSize/2f,
                        });
                    VoxelMaterialType type = (VoxelMaterialType)random.NextInt(0, 2);
                    if (type == VoxelMaterialType.STANDARD)
                    {
                        voxelWorld.AddVoxel(
                            voxelSize,
                            new StandardMaterialData
                            {
                                albedo = RandColor(),
                                specular = RandSpecular(),
                                emission = RandColor() * random.NextFloat(0f, 1f),
                                smoothness = 0.5f,
                                metallic = 0.2f,
                                ior = 0.6f
                            },
                            (uint)(Time.time * 1000)
                        );
                    }
                    else
                    {
                        voxelWorld.AddVoxel(
                            voxelSize,
                            new GlassMaterialData
                            {
                                albedo = RandColor(),
                                emission = RandColor() * random.NextFloat(0f, 1.1f),
                                ior = random.NextFloat(1.0f, 2.8f),
                                roughness = random.NextFloat(0f, 0.5f),
                                extinctionCoeff = 0,
                                flatShading = 0,
                            },
                            (uint)(Time.time * 1000)
                        );
                    }
                }
                verletWorld.worldGravity = gravity;
                // build physics hash grid
                VerletPhysicsHashGrid hashGrid = new VerletPhysicsHashGrid(100000, Allocator.TempJob);
                verletPhysicsHasher.world = verletWorld;
                verletPhysicsHasher.cells = hashGrid.cells.AsParallelWriter();
                verletPhysicsHasher.cellSize = cellSize;
                verletPhysicsHasher.Schedule(verletWorld.ParticleCount, 64).Complete();
                // collisions
                verletPhysicsCollisions.grid = hashGrid;
                verletPhysicsCollisions.dt = simDt;
                verletPhysicsCollisions.cellSize = cellSize;
                verletPhysicsCollisions.world = verletWorld;
                verletPhysicsCollisions.subSteps = subSteps;
                verletPhysicsCollisions.Schedule(verletWorld.ParticleCount, 64).Complete();
                // tick physics
                /*VerletPhysicsCollisionsBruteJob physicsJob = new VerletPhysicsCollisionsBruteJob
                {
                    world = verletWorld,
                    simDt = simDt,
                    subSteps = subSteps
                };
                physicsJob.Schedule().Complete();*/
                hashGrid.Dispose();
            }

            // assemble renderer data
            VerletPhysicsRendererAssembler vrendererAssembler = new VerletPhysicsRendererAssembler(voxelWorld.StandardVoxels, voxelWorld.GlassVoxels, Allocator.TempJob);
            physicsToRendererAssemblerJob.verletWorld = verletWorld;
            physicsToRendererAssemblerJob.voxelWorld = voxelWorld;
            physicsToRendererAssemblerJob.standardMaterialAssembly = vrendererAssembler.standardMaterialAssembly.AsParallelWriter();
            physicsToRendererAssemblerJob.glassMaterialAssembly = vrendererAssembler.glassMaterialAssembly.AsParallelWriter();
            physicsToRendererAssemblerJob.random = random;
            physicsToRendererAssemblerJob.Schedule(verletWorld.ParticleCount, 64).Complete();

            // convert to renderer
            VerletPhysicsRenderer vrenderer = new VerletPhysicsRenderer(voxelWorld.StandardVoxels, voxelWorld.GlassVoxels, Allocator.TempJob);
            VerletPhysicsToRendererConverterJob converterJob = new VerletPhysicsToRendererConverterJob
            {
                renderer = vrenderer,
                rendererAssembler = vrendererAssembler,
            };
            converterJob.Schedule(verletWorld.ParticleCount, 64).Complete();

            vrendererAssembler.Dispose();

            count = verletWorld.ParticleCount;
            lastTick = time;
            // copy data to instanced renderer
            return vrenderer;
        }        

    }



    public struct VerletPhysicsHashGrid : System.IDisposable
    {
        public NativeMultiHashMap<int3, int> cells;
        Allocator allocator;

        public VerletPhysicsHashGrid(int maxCapacity, Allocator a)
        {
            allocator = a;
            cells = new NativeMultiHashMap<int3, int>(maxCapacity, a);
        }

        public void Dispose()
        {
            if (cells.IsCreated)
            {
                cells.Dispose();
            }
        }

        public NativeArray<int3> Hashes(Allocator a)
        {
            return cells.GetKeyArray(a);
        }
    }

    [BurstCompile]
    public struct VerletPhysicsHasherJob : IJobParallelFor
    {
        [ReadOnly] public VerletPhysicsWorld world;
        [ReadOnly] public double cellSize;
        public NativeMultiHashMap<int3, int>.ParallelWriter cells;
        public void Execute(int i)
        {
            //for (int i = 0; i < world.ParticleCount; i++)
            {
                VerletParticle p = world.GetParticle(i);
                int3 hash = p.Hash(cellSize);
                cells.Add(hash, i);
            }
        }
    }

    [BurstCompile]
    public struct VerletPhysicsCollisionsJob : IJobParallelFor
    {
        [ReadOnly] public int subSteps;
        [ReadOnly] public double cellSize;
        [ReadOnly] public double dt;
        [ReadOnly] public VerletPhysicsHashGrid grid;
        [NativeDisableParallelForRestriction] public VerletPhysicsWorld world;

        public void Execute(int index)
        {
            //VerletParticle particle = world.GetParticle(index);
            double substepRat = 1.0 / subSteps;
            VerletParticle particle = world.GetParticle(index);
            if (particle.freeze) return;
            int3 cell = particle.Hash(cellSize);
            int checkSize = 2;//(int)math.floor(particle.radius / cellSize) * 2;
            NativeList<VerletParticle> neighbors = Neighbors(cell, checkSize, index);
            for (int s = 0; s < subSteps; s++)
            {
                particle.accelerate(world.worldGravity);
                particle = CheckCollision(particle, neighbors, substepRat);
                particle.update(dt * substepRat);
            }

            world.SetParticle(index, particle);
            neighbors.Dispose();
        }

        /// <summary>
        /// Pre Condition: neighbors does not contain p
        /// </summary>
        /// <param name="p"></param>
        /// <param name="pid"></param>
        /// <param name="neighbors"></param>
        /// <param name="substepRatio"></param>
        /// <returns></returns>
        VerletParticle CheckCollision(VerletParticle p, NativeList<VerletParticle> neighbors, double substepRatio)
        {
            foreach(VerletParticle n in neighbors)
            {
                double3 vecToN = n.pos_current - p.pos_current;
                double dis = math.length(vecToN);
                double leastDis = p.radius + n.radius;
                if (dis < leastDis)
                {
                    double3 offset = -vecToN / dis * math.abs((leastDis - dis));
                    if (!n.freeze) offset *= 0.5;
                    p.pos_current += offset;
                    //p.vel += -vecToN * substepRatio;
                    //p.vel = (p.mass * p.vel + n.mass * n.vel - n.mass * n.vel) / p.mass * substepRatio;
                }
            }
            return p;
        }

        NativeList<VerletParticle> Neighbors(int3 cell, int3 checkSize, int pid)
        {
            NativeList<VerletParticle> result = new NativeList<VerletParticle>(Allocator.Temp);
            for(int x = -checkSize.x/2; x <= checkSize.x/2; x++)
            {
                for (int y = -checkSize.y / 2; y <= checkSize.y / 2; y++)
                {
                    for (int z = -checkSize.z / 2; z <= checkSize.z / 2; z++)
                    {
                        int3 current = cell + new int3(x, y, z);
                        if (grid.cells.ContainsKey(current))
                        {
                            foreach (var item in grid.cells.GetValuesForKey(current))
                            {
                                if (pid == item) continue;
                                result.Add(world.GetParticle(item));
                            }
                        }
                    }
                }
            }
            return result;
        }
    }

    [BurstCompile]
    public struct VerletPhysicsCollisionsBruteJob : IJob
    {
        [ReadOnly] public double simDt;
        [ReadOnly] public int subSteps;
        public VerletPhysicsWorld world;

        public void Execute()
        {
            for (int s = 0; s < subSteps; s++)
            {
                ApplyGravity();
                SolveCollisions();
                UpdatePositions(simDt / subSteps);
            }
        }

        void ApplyGravity()
        {
            for(int i = 0; i < world.ParticleCount; i++)
            {
                VerletParticle p = world.GetParticle(i);
                if (p.freeze) continue;
                p.accelerate(world.worldGravity);
                world.SetParticle(i, p);
            }
        }

        void UpdatePositions(double dt)
        {
            for (int i = 0; i < world.ParticleCount; i++)
            {
                VerletParticle p = world.GetParticle(i);
                if (p.freeze) continue;
                p.update(dt);
                world.SetParticle(i, p);
            }
        }

        void SolveCollisions()
        {
            for (int i = 0; i < world.ParticleCount; i++)
            {
                VerletParticle a = world.GetParticle(i);
                for (int j = 0; j < world.ParticleCount; j++)
                {
                    if (i == j) continue;
                    VerletParticle b = world.GetParticle(j);

                    double3 collision_axis = a.pos_current - b.pos_current;
                    double dist = math.length(collision_axis);
                    double minSep = a.radius + b.radius;

                    if (dist < minSep)
                    {
                        double3 n = collision_axis / dist;
                        double delta = minSep - dist;
                        double mult = 0.5;
                        if (!a.freeze)
                            a.pos_current += mult * delta * n;
                        if (!b.freeze)
                            b.pos_current -= mult * delta * n;
                    }

                    world.SetParticle(j, b);
                }
                world.SetParticle(i, a);
            }
        }
    }

    [BurstCompile]
    public struct VerletPhysicsToRendererAssemblerJob : IJobParallelFor
    {
        [ReadOnly] public VerletPhysicsWorld verletWorld;
        [ReadOnly] public VoxelWorld voxelWorld;
        [ReadOnly] public Unity.Mathematics.Random random;
        [NativeDisableParallelForRestriction]
        public NativeList<StandardVoxelAssembledData>.ParallelWriter standardMaterialAssembly;
        [NativeDisableParallelForRestriction]
        public NativeList<GlassVoxelAssembledData>.ParallelWriter glassMaterialAssembly;

        public void Execute(int index)
        {
            VerletParticle p = verletWorld.GetParticle(index);
            float3 pos = new float3((float)p.pos_current.x, (float)p.pos_current.y, (float)p.pos_current.z);

            Voxel v = voxelWorld.GetVoxel(index);
            float4x4 trs = float4x4.TRS(pos, quaternion.identity, new float3(1, 1, 1) * v.size);


            if (v.type == VoxelMaterialType.STANDARD)
            {
                standardMaterialAssembly.AddNoResize(new StandardVoxelAssembledData{
                    material= voxelWorld.GetStandardMaterial(v.id),
                    trs = trs
                });
            }
            else if (v.type == VoxelMaterialType.GLASS)
            {
                glassMaterialAssembly.AddNoResize(new GlassVoxelAssembledData
                {
                    material = voxelWorld.GetGlassMaterial(v.id),
                    trs = trs
                });
            }
        }
    }

    [BurstCompile]
    public struct VerletPhysicsToRendererConverterJob : IJobParallelFor
    {
        [ReadOnly] public VerletPhysicsRendererAssembler rendererAssembler;
        [NativeDisableParallelForRestriction]
        public VerletPhysicsRenderer renderer;

        public void Execute(int index)
        {
            if (index < rendererAssembler.standardMaterialAssembly.Length)
            {
                StandardVoxelAssembledData assembly = rendererAssembler.standardMaterialAssembly[index];
                renderer.standardMatrices[index] = assembly.trs;
                renderer.standardMaterialData[index] = assembly.material;
            }
            else if (index < rendererAssembler.standardMaterialAssembly.Length + rendererAssembler.glassMaterialAssembly.Length)
            {
                int idx = index - rendererAssembler.standardMaterialAssembly.Length;
                if (idx < 0 || idx >= rendererAssembler.glassMaterialAssembly.Length) return;
                GlassVoxelAssembledData assembly = rendererAssembler.glassMaterialAssembly[idx];
                renderer.glassMatrices[idx] = assembly.trs;
                renderer.glassMaterialData[idx] = assembly.material;
            }
        }
    }
}