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

        VerletPhysicsSimulator simulator;
        VerletPhysicsWorld world;

        private void Start()
        {
            StartCoroutine(RunPhysics());
        }

        private void OnDestroy()
        {
            world.Dispose();
        }

        IEnumerator RunPhysics()
        {
            world = new VerletPhysicsWorld(gravity);
            simulator = new VerletPhysicsSimulator();
            while (true)
            {
                world.worldGravity = gravity;
                simulator.dt = (double)simDt;
                JobHandle handle = simulator.Schedule(world.ParticleCount, 64);
                yield return new WaitUntil(() => handle.IsCompleted);
                yield return new WaitForSeconds(simDt);
            }
        }

    }

    // https://en.wikipedia.org/wiki/Verlet_integration
    [System.Serializable]
    public struct VerletParticle
    {
        public double3 pos;
        public double3 vel;
        public double3 acc;

        public double mass; // 1kg
        public double drag; // rho*C*Area – simplified drag for this example

        public VerletParticle(double _mass, double _drag)
        {
            this.pos = 0;
            this.vel = 0;
            this.acc = 0;
            this.mass = _mass;
            this.drag = _drag;
        }

        public void update(double dt)
        {
            double3 new_pos = pos + vel * dt + acc * (dt * dt * 0.5);
            double3 new_acc = dragForces(); // only needed if acceleration is not constant
            double3 new_vel = vel + (acc + new_acc) * (dt * 0.5);
            pos = new_pos;
            vel = new_vel;
            acc = new_acc;
        }

        double3 dragForces()
        {
           // double3 grav_acc = new double3 ( 0.0, 0.0, -9.81 ); // 9.81 m/s² down in the z-axis
            double3 drag_force = 0.5 * drag * (vel * vel); // D = 0.5 * (rho * C * Area * vel^2)
            double3 drag_acc = drag_force / mass; // a = F/m
            return /*grav_acc*/ - drag_acc;
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

}