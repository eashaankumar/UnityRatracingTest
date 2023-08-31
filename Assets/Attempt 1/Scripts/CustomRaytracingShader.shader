Shader "Unlit/CustomRaytracingShader"
{
    Properties
    {
        //_MainTex ("Texture", 2D) = "white" {}
        _Color("Color", Color) = (1,1,1,1)
        _Energy("Energy", Float) = 1
    }
        SubShader
    {
        Pass
        {
            Name "Test"
            Tags{ "LightMode" = "RayTracing" }

            HLSLPROGRAM

            #include "UnityRaytracingMeshUtils.cginc"
            #include "RayPayload.hlsl"

            #pragma raytracing test 

        //Texture2D<float4> _MainTex;
        //SamplerState sampler__MainTex;
        //float4 _MainTex_ST; // This is Tiling(.xy)/Offset(.zw) option in the Material inspector.
            float4 _Color;
            float _Energy;
            RaytracingAccelerationStructure g_SceneAccelStruct;

            // https://github.com/INedelcu/RayTracingShader_VertexAttributeInterpolation/tree/autodesk-interactive-with-ray-tracing
            // https://forum.unity.com/threads/dxr-raytracing-effect-from-scratch.794928/ (for recursive rays)

            struct AttributeData
            {
                float2 barycentrics;
            };

            struct Vertex
            {
                float3 normal;
                float2 uv;
            };

            Vertex FetchVertex(uint vertexIndex)
            {
                Vertex v;
                v.normal = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeNormal);
                v.uv = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeTexCoord0);
                return v;
            }

            Vertex InterpolateVertices(Vertex v0, Vertex v1, Vertex v2, float3 barycentrics)
            {
                Vertex v;
                #define INTERPOLATE_ATTRIBUTE(attr) v.attr = v0.attr * barycentrics.x + v1.attr * barycentrics.y + v2.attr * barycentrics.z
                INTERPOLATE_ATTRIBUTE(normal);
                INTERPOLATE_ATTRIBUTE(uv);
                return v;
            }

            [shader("closesthit")]
            void ClosestHitMain(inout RayPayload payload, in AttributeData attribs)
            {
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());

                Vertex v0, v1, v2;
                v0 = FetchVertex(triangleIndices.x);
                v1 = FetchVertex(triangleIndices.y);
                v2 = FetchVertex(triangleIndices.z);

                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
                Vertex v = InterpolateVertices(v0, v1, v2, barycentricCoords);
                
                // reflect/refract
                float3 worldNormal = normalize(mul(v.normal, (float3x3)WorldToObject()));
                float3 worldRayOrigin = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
                if (payload.bounceIndex < 2)
                {
                    float3 reflectedRay = reflect(WorldRayDirection(), worldNormal);

                    RayDesc ray;
                    ray.Origin = worldRayOrigin + 0.01f * reflectedRay;
                    ray.Direction = reflectedRay;
                    ray.TMin = 0;
                    ray.TMax = 1000;

                    RayPayload payload2;
                    payload2.color = float4(0, 0, 0, 0);
                    payload2.energy = 0;
                    payload2.bounceIndex = payload.bounceIndex + 1;

                    uint missShaderIndex = 0;
                    //TraceRay(g_SceneAccelStruct, 0, 0xFF, 0, 1, missShaderIndex, ray, payload2);

                    // fill payload
                    float3 color = _Color;
                    payload.color.xyz = color + payload2.color.xyz;
                    payload.energy = _Energy + payload2.energy;
                }
                else 
                {
                    float3 color = _Color;
                    payload.color.xyz = color;
                    payload.energy = _Energy;
                }
            }

            ENDHLSL
        }
    }
}
