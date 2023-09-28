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

            public string BaseFilePathSep
            {
                get
                {
                    return targetFolder + "\\" + datasetName + Dataset.SEP;
                }
            }

        }
        [SerializeField]
        DatasetInfo info;
        

        int id;
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

        private void Awake()
        {
            id = 0;
            info.datasetName = info.datasetName.Replace(SEP + "", "");
        }

        public void AddData(ref RenderTexture noisy, ref RenderTexture normals, ref RenderTexture depth,
                            ref RenderTexture albedo, ref RenderTexture shape, ref RenderTexture emission,
                            ref RenderTexture material, ref RenderTexture converged)
        {
            string baseFilePathSep = info.BaseFilePathSep;
            SaveTexture(ref noisy, baseFilePathSep, "noisy");
            SaveTexture(ref normals, baseFilePathSep, "normals");
            SaveTexture(ref depth, baseFilePathSep, "depth");
            SaveTexture(ref albedo, baseFilePathSep, "albedo");
            SaveTexture(ref shape, baseFilePathSep, "shape");
            SaveTexture(ref emission, baseFilePathSep, "emission");
            SaveTexture(ref material, baseFilePathSep, "material");
            SaveTexture(ref converged, baseFilePathSep, "converged");
            id++;
        }

        void SaveTexture(ref RenderTexture rt, string baseFilePathSep, string name)
        {
            byte[] bytes = toTexture2D(ref rt).EncodeToJPG();
            File.WriteAllBytes(baseFilePathSep + name + SEP + id + ".jpg", bytes);
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