using BarelyFunctional.Interfaces;
using BarelyFunctional.Structs;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace BarelyFunctional.Structs
{
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

        public void update(double dt, double3 gravityAcc)
        {
            double3 new_pos = pos + vel * dt + acc * (dt * dt * 0.5);
            double3 new_acc = gravityAcc + dragAcc();
            double3 new_vel = vel + (acc + new_acc) * (dt * 0.5);
            pos = new_pos;
            vel = new_vel;
            acc = new_acc;
        }

        double3 dragAcc()
        {
            // double3 grav_acc = new double3 ( 0.0, 0.0, -9.81 ); // 9.81 m/s² down in the z-axis
            double3 drag_force = 0.5 * drag * (vel * vel); // D = 0.5 * (rho * C * Area * vel^2)
            double3 drag_acc = drag_force / mass; // a = F/m
            return /*grav_acc*/ -drag_acc;
        }
    }

    public struct Data
    {
        public float3 color;
        public float emission;
    };

    public struct VerletPhysicsRenderer : System.IDisposable
    {
        public NativeArray<Data> data;
        public NativeArray<Matrix4x4> matrices;

        public VerletPhysicsRenderer(int count, Allocator alloc)
        {
            data = new NativeArray<Data>(count, alloc);
            matrices = new NativeArray<Matrix4x4>(count, alloc);
        }

        public void Dispose()
        {
            if (data.IsCreated) data.Dispose();
            if (matrices.IsCreated) matrices.Dispose();
        }
    }
}

namespace BarelyFunctional.Interfaces
{
    public interface IVerletPhysicsRenderer
    {
        public void SetRenderer(VerletPhysicsRenderer readyToBeCopied);
    }
}
