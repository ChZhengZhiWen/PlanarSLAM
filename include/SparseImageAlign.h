#ifndef Planar_SLAM_SPARSE_IMAGE_ALIGN_
#define Planar_SLAM_SPARSE_IMAGE_ALIGN_

#include "Common.h"
#include "NLSSolver.h"
#include "Frame.h"

// 稀疏直接法求解器
// 请注意SVO的直接法用了一种逆向的奇怪解法，它的雅可比是在Ref中估计而不是在Current里估计的，所以迭代过程中雅可比是不动的

namespace Planar_SLAM {

    /// Optimize the pose of the frame by minimizing the photometric error of feature patches.
    class SparseImgAlign : public NLLSSolver<6, SE3f> {
        static const int patch_halfsize_ = 2;
        static const int patch_size_ = 2 * patch_halfsize_;
        static const int patch_area_ = patch_size_ * patch_size_;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        cv::Mat resimg_;

        /**
         * @brief constructor
         * @param[in] n_levels total pyramid level
         * @param[in] min_level minimum levels
         * @param[in] n_iter iterations
         * @param[in] methos GaussNewton or LevernbergMarquardt
         * @param[in] display display the residual image
         * @param[in] verbose output the inner computation information
         */
        SparseImgAlign(
                int n_levels,
                int min_level,
                float m_chi2,
                float m_nMeans,
                int n_iter = 10,
                Method method = GaussNewton,
                bool display = false,
                bool verbose = false);

        /**
         * 计算 ref 和 current 之间的运动
         * @brief compute the relative motion between ref frame and current frame
         * @param[in] ref_frame the reference
         * @param[in] cur_frame the current frame
         * @param[out] TCR motion from ref to current
         */
        size_t run(
                Frame *ref_frame,
                Frame *cur_frame,
                SE3f &TCR
        );

        void mat2SE3(cv::Mat org,SE3f &dst);

        /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
        /// at the converged state.
        Matrix<float, 6, 6> getFisherInformation();

    protected:
        Frame *ref_frame_;              //!< reference frame, has depth for gradient pixels.
        Frame *cur_frame_;              //!< only the image is known!
        int level_;                     //!< current pyramid level on which the optimization runs.
        bool display_;                  //!< display residual image.
        int max_level_;                 //!< coarsest pyramid level for the alignment.
        int min_level_;                 //!< finest pyramid level for the alignment.

        float sparseChi2;
        float sparseNMeans;

        // cache:
        Matrix<float, 6, Dynamic, ColMajor> jacobian_cache_;    // 雅可比矩阵，这个东西是固定下来的

        /// Cached values for all pixels corresponding to a list of feature-patches
        struct Cache
        {
            Matrix<float, 6, Dynamic, ColMajor> jacobian;  // cached jacobian
            cv::Mat ref_patch;  // cached patch intensity values (with subpixel precision)
            std::vector<bool> visible_fts; // mask of visible features
            std::map<int,std::map<int,std::vector<float>>> seg_ref_patch;

            Cache() {} // default constructor
            Cache( size_t num_fts, int patch_area ) // constructor
            {
                // resize cache variables according to the maximum number of incoming features
                ref_patch = cv::Mat(num_fts, patch_area, CV_32F);
                jacobian.resize(Eigen::NoChange, num_fts*patch_area);
                visible_fts.resize(num_fts, false); // TODO: should it be reset at each level?
            }
        };
        Cache pt_cache_;
        Cache seg_cache_;
        std::vector<size_t> patch_offset;   // offset for the segment cache

        bool have_ref_patch_cache_;
        cv::Mat ref_patch_cache_;
        std::vector<bool> visible_fts_;
        int normTh = 20;

        int segNum;
        std::vector<Vector2f> xyz_ref_compute;
        std::vector<Vector2f> xyz_ref_computeEnd;
        std::vector<Vector2f> xyz_cur_compute;
        std::vector<Vector2f> xyz_cur_computeEnd;

        std::vector<Vector2f> preCompute;
        std::vector<Vector3f> preCompute3D;
        std::vector<Vector3f> preCompute3DEnd;
        std::vector<Vector2f> preComputeEnd;

        std::map<int,Vector2f> prePointCompute;


        std::map<int,std::vector<Vector2f>> preCompute_map;
        std::map<int,std::vector<Vector3f>> preCompute3D_map;
        std::map<int,std::vector<Vector3f>> preCompute3DEnd_map;
        std::map<int,std::vector<Vector2f>> preComputeEnd_map;
        std::map<int,std::vector<Vector2f>> xyz_ref_compute_map;
        std::map<int,std::vector<Vector2f>> xyz_ref_computeEnd_map;
        std::map<int,std::vector<Vector2f>> xyz_cur_compute_map;
        std::map<int,std::vector<Vector2f>> xyz_cur_computeEnd_map;

        // 在ref中计算雅可比
        void precomputeReferencePatches();
        void precomputeReferencePatches_zzw();

        // 派生出来的虚函数
        virtual float computeResiduals(const SE3f &model, bool linearize_system, bool compute_weight_scale = false);
        virtual float computeResiduals_zzw(const SE3f &model, bool linearize_system, bool compute_weight_scale = false);

        void computeGaussNewtonParamsPoints(
                const SE3f &T_cur_from_ref, bool linearize_system, bool compute_weight_scale,
                Cache &cache, Matrix<float, 6, 6> &H, Matrix<float, 6, 1> &Jres,
                std::vector<float> &errors, float &chi2);

        void computeGaussNewtonParamsSegments(const SE3f &T_cur_from_ref, bool linearize_system, bool compute_weight_scale,
                                              Cache &cache, Matrix<float, 6, 6> &H, Matrix<float, 6, 1> &Jres,
                                              std::vector<float> &errors, float &chi2);

        void precomputeGaussNewtonParamsPoints(Cache &cache);
        void precomputeGaussNewtonParamsSegments(Cache &cache);

        virtual int solve();

        virtual void update(const ModelType &old_model, ModelType &new_model);

        virtual void startIteration();

        virtual void finishIteration();

        // *************************************************************************************
        // 一些固定的雅可比
        // xyz 到 相机坐标 的雅可比，平移在前
        // 这里已经取了负号，不要再取一遍！
        inline Eigen::Matrix<float, 2, 6> JacobXYZ2Cam(const Vector3f &xyz) {
            Eigen::Matrix<float, 2, 6> J;
            const float x = xyz[0];
            const float y = xyz[1];
            const float z_inv = 1. / xyz[2];
            const float z_inv_2 = z_inv * z_inv;

            J(0, 0) = -z_inv;           // -1/z
            J(0, 1) = 0.0;              // 0
            J(0, 2) = x * z_inv_2;        // x/z^2
            J(0, 3) = y * J(0, 2);      // x*y/z^2
            J(0, 4) = -(1.0 + x * J(0, 2)); // -(1.0 + x^2/z^2)
            J(0, 5) = y * z_inv;          // y/z

            J(1, 0) = 0.0;              // 0
            J(1, 1) = -z_inv;           // -1/z
            J(1, 2) = y * z_inv_2;        // y/z^2
            J(1, 3) = 1.0 + y * J(1, 2); // 1.0 + y^2/z^2
            J(1, 4) = -J(0, 3);       // -x*y/z^2
            J(1, 5) = -x * z_inv;         // x/z
            return J;
        }

        inline size_t setupSampling(size_t patch_size, Vector2f &dif,Vector2f spx,Vector2f epx,float length)
        {
            // complete sampling of the segment surroundings,
            // with minimum overlap of the square patches
            // if segment is horizontal or vertical, N is seg_length/patch_size
            // if the segment has angle theta, we need to correct according to the distance from center to unit-square border: *corr
            // scale (pyramid level) is accounted for later
            dif = epx - spx; // difference vector from start to end point
            double tan_dir = std::min(fabs(dif[0]),fabs(dif[1])) / std::max(fabs(dif[0]),fabs(dif[1]));
            double sin_dir = tan_dir / sqrt( 1.0+tan_dir*tan_dir );//1+tanx^2=secx²
            double correction = 2.0 * sqrt( 1.0 + sin_dir*sin_dir );


//            cout<<"dif:"<<dif[0]<<","<<dif[1]<<"    length:"<<length<<endl;
//            cout<<"tan_dir:"<<tan_dir<<"   sin_dir:"<<sin_dir<<"   correction:"<<correction<<endl;
//            return std::max( 1.0, length / (2.0*patch_size*correction) );
            return std::max( 1.0, length / (patch_size*correction) );
            // If length is very low the segment approaches a point and the minimum 1 sample is taken (the central point)
        }

    };

}// namespace


#endif