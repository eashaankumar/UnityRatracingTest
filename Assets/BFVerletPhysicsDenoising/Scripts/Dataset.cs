using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace BarelyFunctional.Renderer.Denoiser.DataGeneration
{
    public class Dataset : MonoBehaviour
    {
        [System.Serializable]
        public struct DatasetInfo
        {
            public string targetFolder; // C:\Users\seana\OneDrive\Pictures\PathTracing\data
            public int width;
            public int height;
            public string datasetName; // DenoisingDataset
            public int convergence;
            public uint samples;

            [Range(1, 100)]
            public uint bounceCountOpaque;

            [Range(1, 100)]
            public uint bounceCountTransparent;

        }
        [SerializeField]
        DatasetInfo info;
        

        static readonly char SEP = '-';

        public int PixelWidth
        {
            get
            {
                return info.width;
            }
        }

        public int PixelHeight
        {
            get { return info.height; }
        }

        public int Convergence
        {
            get { return info.convergence; }
        }

        public uint BounceCountOpaque
        {
            get { return info.bounceCountOpaque; }
        }

        public uint BounceCountTransparent
        {
            get { return info.bounceCountTransparent; }
        }

        public uint Samples
        {
            get
            {
                return info.samples;
            }
        }

        private void Awake()
        {
            info.datasetName = info.datasetName.Replace(SEP + "", "");
        }

        public void AddData(int id, ref RenderTexture noisy, ref RenderTexture normals, ref RenderTexture depth,
                            ref RenderTexture albedo, ref RenderTexture shape, ref RenderTexture emission,
                            ref RenderTexture specular, ref RenderTexture converged)
        {
            string baseFilePath = info.targetFolder + "\\" + id + "\\";
            SaveTexture(ref noisy, baseFilePath, "noisy", id);
            SaveTexture(ref normals, baseFilePath, "normals", id);
            SaveTexture(ref depth, baseFilePath, "depth", id);
            SaveTexture(ref albedo, baseFilePath, "albedo", id);
            SaveTexture(ref shape, baseFilePath, "shape", id);
            SaveTexture(ref emission, baseFilePath, "emission", id);
            SaveTexture(ref specular, baseFilePath, "specular", id);
            SaveTexture(ref converged, baseFilePath, "converged", id);
        }

        void SaveTexture(ref RenderTexture rt, string baseFilePathSep, string name, int id)
        {
            byte[] bytes = toTexture2D(ref rt).EncodeToJPG();
            bool exists = System.IO.Directory.Exists(baseFilePathSep);
            if (!exists)
                System.IO.Directory.CreateDirectory(baseFilePathSep);
            File.WriteAllBytes(baseFilePathSep + info.datasetName + Dataset.SEP + name + SEP + id + ".jpg", bytes);
        }

        Texture2D toTexture2D(ref RenderTexture rTex)
        {
            Texture2D tex = new Texture2D(info.width, info.height, TextureFormat.RGB24, false);
            tex.filterMode = FilterMode.Point;
            RenderTexture.active = rTex;
            tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
            tex.Apply();
            return tex;
        }
    }
}