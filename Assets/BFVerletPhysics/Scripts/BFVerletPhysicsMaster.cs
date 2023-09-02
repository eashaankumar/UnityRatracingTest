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

        public int count;

        VerletPhysicsSimulator simulatorJob;
        VerletPhysicsWorld verletWorld;
        VoxelWorld voxelWorld;
        VerletPhysicsToRendererConverter physicsToRendererConverterJob;

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
            voxelWorld = new VoxelWorld(Allocator.Persistent);
            simulatorJob = new VerletPhysicsSimulator();
            physicsToRendererConverterJob = new VerletPhysicsToRendererConverter();
            random = new Unity.Mathematics.Random(1224214);
        }

        public VerletPhysicsRenderer Tick()
        {
            float time = Time.time;
            JobHandle handle;
            if (time - lastTick >= simDt)
            {
                verletWorld.AddParticle(
                    new VerletParticle
                    {
                        drag = 0,
                        mass = 1.0,
                        vel = random.NextDouble3Direction() * 5
                    });
                voxelWorld.AddVoxel(
                    new Voxel
                    (
                        new Structs.Color(random.NextFloat(), random.NextFloat(), random.NextFloat()),
                        random.NextFloat()
                    ));
                verletWorld.worldGravity = gravity;
                simulatorJob.dt = simDt;
                simulatorJob.world = verletWorld;
                handle = simulatorJob.Schedule(verletWorld.ParticleCount, 64);
                handle.Complete();
            }

            // convert to renderer
            VerletPhysicsRenderer vrenderer = new VerletPhysicsRenderer(verletWorld.ParticleCount, Allocator.TempJob);
            physicsToRendererConverterJob.verletWorld = verletWorld;
            physicsToRendererConverterJob.voxelWorld = voxelWorld;
            physicsToRendererConverterJob.renderer = vrenderer;
            physicsToRendererConverterJob.random = random;
            handle = physicsToRendererConverterJob.Schedule(verletWorld.ParticleCount, 64);
            handle.Complete();

            count = verletWorld.ParticleCount;
            lastTick = time;
            // copy data to instanced renderer
            return vrenderer;
        }        

    }

    public struct VoxelWorld : System.IDisposable
    {
        NativeList<Voxel> voxels;

        public VoxelWorld(Allocator a)
        {
            voxels = new NativeList<Voxel>(a);
        }

        public void Dispose()
        {
            if (voxels.IsCreated) voxels.Dispose();    
        }

        public Voxel GetVoxel(int i)
        {
            return voxels[i];
        }

        public void AddVoxel(Voxel v)
        {
            if(voxels.IsCreated) voxels.Add(v);
        }
    }

    public struct VerletPhysicsWorld : System.IDisposable
    {
        public NativeList<VerletParticle> particles;
        public double3 worldGravity;
        bool isCreated;

        public bool IsCreated
        {
            get { return isCreated; }
        }

        public VerletPhysicsWorld(double3 g, Allocator a)
        {
            particles = new NativeList<VerletParticle>(a);
            worldGravity = g;
            isCreated = true;
        }

        public void AddParticle(VerletParticle p)
        {
            if (!particles.IsCreated) return;
            particles.Add(p);
        }

        public int ParticleCount
        {
            get
            {
                return particles.IsCreated ? particles.Length : 0;
            }
        }

        public VerletParticle GetParticle(int i)
        {
            return particles[i];
        }

        public void SetParticle(int i, VerletParticle p)
        {
            particles[i] = p;
        }

        public void Dispose()
        {
            if (particles.IsCreated) particles.Dispose();
            isCreated = false;
        }
    }

    [BurstCompile]
    public struct VerletPhysicsSimulator : IJobParallelFor
    {
        [ReadOnly] public double dt;
        [NativeDisableParallelForRestriction] public VerletPhysicsWorld world;

        public void Execute(int index)
        {
            //if (!world.IsCreated) return;
            //if (index >= world.ParticleCount) return;

            VerletParticle particle = world.GetParticle(index);
            //particle.acc += world.worldGravity;
            particle.update(dt, world.worldGravity);
            world.SetParticle(index, particle);
        }
    }

    [BurstCompile]
    public struct VerletPhysicsToRendererConverter : IJobParallelFor
    {
        [ReadOnly] public VerletPhysicsWorld verletWorld;
        [ReadOnly] public VoxelWorld voxelWorld;
        [ReadOnly] public Unity.Mathematics.Random random;
        public VerletPhysicsRenderer renderer;

        public void Execute(int index)
        {
            Voxel v = voxelWorld.GetVoxel(index);
            renderer.data[index] = new Data()
            {
                color = v.color.Float3(),
                emission = v.glow / 255f, 
            };

            VerletParticle p = verletWorld.GetParticle(index);
            float3 pos = new float3((float)p.pos.x, (float)p.pos.y, (float)p.pos.z);
            renderer.matrices[index] = float4x4.TRS(pos, quaternion.identity, new float3(1, 1, 1));
        }
    }

}