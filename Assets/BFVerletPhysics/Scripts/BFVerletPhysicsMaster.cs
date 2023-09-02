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
        VerletPhysicsWorld world;
        VerletPhysicsToRendererConverter physicsToRendererConverterJob;

        float lastTick;

        Unity.Mathematics.Random random;

        private void OnDestroy()
        {
            world.Dispose();
        }

        private void Awake()
        {
            world = new VerletPhysicsWorld(gravity);
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
                world.AddParticle(
                    new VerletParticle 
                    { 
                        drag = 0, 
                        mass = 1.0, 
                        vel = new double3(UnityEngine.Random.value * 5 * 2 - 1, UnityEngine.Random.value * 10, UnityEngine.Random.value * 5 * 2 - 1) 
                    });
                world.worldGravity = gravity;
                simulatorJob.dt = simDt;
                simulatorJob.world = world;
                handle = simulatorJob.Schedule(world.ParticleCount, 64);
                handle.Complete();
            }

            // convert to renderer
            VerletPhysicsRenderer vrenderer = new VerletPhysicsRenderer(world.ParticleCount, Allocator.TempJob);
            physicsToRendererConverterJob.world = world;
            physicsToRendererConverterJob.renderer = vrenderer;
            physicsToRendererConverterJob.random = random;
            handle = physicsToRendererConverterJob.Schedule(world.ParticleCount, 64);
            handle.Complete();

            count = world.ParticleCount;
            lastTick = time;
            // copy data to instanced renderer
            return vrenderer;
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

        public VerletPhysicsWorld(double3 g)
        {
            particles = new NativeList<VerletParticle>(Allocator.Persistent);
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
        [ReadOnly] public VerletPhysicsWorld world;
        [ReadOnly] public Unity.Mathematics.Random random;
        public VerletPhysicsRenderer renderer;

        public void Execute(int index)
        {
            renderer.data[index] = new Data() 
            { 
                color = new float3(random.NextFloat(0f, 1f), random.NextFloat(0f, 1f), random.NextFloat(0f, 1f)), emission = random.NextFloat(0f, 1f) 
            };

            VerletParticle p = world.GetParticle(index);
            float3 pos = new float3((float)p.pos.x, (float)p.pos.y, (float)p.pos.z);
            renderer.matrices[index] = float4x4.TRS(pos, quaternion.identity, new float3(1, 1, 1));
        }
    }

}