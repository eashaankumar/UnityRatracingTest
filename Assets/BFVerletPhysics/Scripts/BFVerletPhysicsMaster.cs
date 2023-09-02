using BarelyFunctional.AbstractClasses;
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
        [SerializeField]
        float simDt;
        [SerializeField]
        VerletPhysicsParticleRenderer verletPhysicsRenderer;

        VerletPhysicsSimulator simulatorJob;
        VerletPhysicsWorld world;
        VerletPhysicsToRendererConverter physicsToRendererConverterJob;
        VerletPhysicsRenderer vrenderer;

        private void Start()
        {
            StartCoroutine(RunPhysics());
        }

        private void OnDestroy()
        {
            world.Dispose();
            vrenderer.Dispose();
        }

        IEnumerator RunPhysics()
        {
            world = new VerletPhysicsWorld(gravity);
            simulatorJob = new VerletPhysicsSimulator();
            physicsToRendererConverterJob = new VerletPhysicsToRendererConverter();
            while (true)
            {
                world.worldGravity = gravity;
                simulatorJob.dt = (double)simDt;
                JobHandle handle = simulatorJob.Schedule(world.ParticleCount, 64);
                yield return new WaitUntil(() => handle.IsCompleted);
                
                // convert to renderer
                vrenderer = new VerletPhysicsRenderer(world.ParticleCount);
                physicsToRendererConverterJob.world = world;
                physicsToRendererConverterJob.renderer = vrenderer;
                handle = physicsToRendererConverterJob.Schedule(world.ParticleCount, 64);
                yield return new WaitUntil(() => handle.IsCompleted);
                // copy data to instanced renderer
                verletPhysicsRenderer.SetRenderer(vrenderer);
                vrenderer.Dispose();
                yield return new WaitForSeconds(simDt);
            }
        }

    }

   

    public struct VerletPhysicsWorld : System.IDisposable
    {
        NativeList<VerletParticle> particles;
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
        public VerletPhysicsWorld world;

        public void Execute(int index)
        {
            if (!world.IsCreated) return;
            if (index >= world.ParticleCount) return;

            VerletParticle particle = world.GetParticle(index);
            particle.acc += world.worldGravity;
            particle.update(dt);
            world.SetParticle(index, particle);
        }
    }

    [BurstCompile]
    public struct VerletPhysicsToRendererConverter : IJobParallelFor
    {
        [ReadOnly] public VerletPhysicsWorld world;
        public VerletPhysicsRenderer renderer;

        public void Execute(int index)
        {
            renderer.data[index] = new Data() { color = new float3(0, 1, 0), emission = 1 };

            VerletParticle p = world.GetParticle(0);
            float3 pos = new float3((float)p.pos.x, (float)p.pos.y, (float)p.pos.z);
            renderer.matrices[index] = float4x4.TRS(pos, quaternion.identity, new float3(1, 1, 1));
        }
    }

}