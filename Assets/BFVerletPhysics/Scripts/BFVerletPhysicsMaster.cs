using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace BarelyFunctional.VerletPhysics
{ 
    public class BFVerletPhysicsMaster : MonoBehaviour
    {

        
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

    public struct VerletPhysicsSimulator : System.IDisposable, IJobParallelFor
    {
        public double3 worldGravity;
        public float dt;

        NativeList<VerletParticle> particles;

        public VerletPhysicsSimulator(float3 g)
        {
            particles = new NativeList<VerletParticle>(Allocator.Persistent);
            worldGravity = g;
            dt = 0;
        }

        public void Dispose()
        {
            if (particles.IsCreated) particles.Dispose();
        }

        public void Execute(int index)
        {
            if (!particles.IsCreated) return;
            if (index >= particles.Length) return;

            VerletParticle particle = particles[index];
            particle.acc += worldGravity;
            particle.update(dt);
            particles[index] = particle;
        }
    }

}