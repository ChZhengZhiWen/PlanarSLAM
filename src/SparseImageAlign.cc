#include "SparseImageAlign.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Feature.h"

namespace Planar_SLAM {
using namespace cv::line_descriptor;

    SparseImgAlign::SparseImgAlign(
            int max_level, int min_level,float m_chi2,float m_nMeans, int n_iter,
            Method method, bool display, bool verbose) :
            display_(display),
            max_level_(max_level),
            min_level_(min_level),
            sparseChi2(m_chi2),
            sparseNMeans(m_nMeans) {
        n_iter_ = n_iter;
        n_iter_init_ = n_iter_;
        method_ = method;
        verbose_ = verbose;
        eps_ = 0.000001;
    }

    size_t SparseImgAlign::run(Frame *ref_frame, Frame *cur_frame, SE3f &TCR) {

        reset();

        if (ref_frame->mvKeys.empty() && ref_frame->mvKeylinesUn.empty()) {
            cout << "SparseImgAlign: no features to track!" << endl;
            return 0;
        }

        ref_frame_ = ref_frame;
        cur_frame_ = cur_frame;

        //ref_patch_cache_：参考帧patch的缓存，即每行一个特征patch的16个像素灰度值
        ref_patch_cache_ = cv::Mat(ref_frame->N, patch_area_, CV_32F);//create  n x 16 matrix
        //雅克比矩阵，每一个特征patch的每个像素对应一个6*1的雅克比
        jacobian_cache_.resize(Eigen::NoChange, ref_patch_cache_.rows * patch_area_);//6 x (n x 16)
        visible_fts_ = vector<bool>(ref_patch_cache_.rows, false);// n x 1

        float total_length = 0;
        for (auto & it : ref_frame_->mvKeylinesUn) {
            Vector2f sp_l;
            sp_l << it.startPointX, it.startPointY;
            Vector2f ep_l;
            ep_l << it.endPointX, it.endPointY;
            float length = (ep_l-sp_l).norm();
            total_length += length;
        }
//
//        for (auto & it : ref_frame_->mLKLine) {
//            Vector2f sp_l;
//            sp_l << it.first[0], it.first[1];
//            Vector2f ep_l;
//            ep_l << it.second[0], it.second[1];
//            float length = (ep_l-sp_l).norm();
//            total_length += length;
//        }
        int max_num_seg_samples = std::ceil( total_length / patch_size_ );

        float edge_total_length = 0;
        for (auto & it : ref_frame_->allPlaneEdgeLine) {
            Vector3f startPoint;
            Vector3f endPoint;
            startPoint[0]=it.first.x;
            startPoint[1]=it.first.y;
            startPoint[2]=it.first.z;
            endPoint[0]=it.second.x;
            endPoint[1]=it.second.y;
            endPoint[2]=it.second.z;
            Vector2f sp_edge = ref_frame_->Camera2Pixel(startPoint);
            Vector2f ep_edge = ref_frame_->Camera2Pixel(endPoint);

            float length = (ep_edge-sp_edge).norm();
            edge_total_length += length;
        }
        int max_num_edge_samples = std::ceil( edge_total_length / patch_size_ );

        //lk光流获得的线段
        float lkSegment_total_length = 0;
        for (auto & it : ref_frame_->mLKLine) {
            Vector2f sp_l;
            sp_l << it.first[0], it.first[1];
            Vector2f ep_l;
            ep_l << it.second[0], it.second[1];
            float length = (ep_l-sp_l).norm();
            lkSegment_total_length += length;
        }
        int max_num_lkSegment_samples = std::ceil( lkSegment_total_length / patch_size_ );

        pt_cache_   = Cache( ref_frame_->N, patch_area_ );
        seg_cache_  = Cache( max_num_seg_samples, patch_area_ );
        edge_cache_ = Cache( max_num_edge_samples * 2, patch_area_ );
        lkSeg_cache_ = Cache( max_num_lkSegment_samples * 2, patch_area_ );

        //Tcr
        //! cv:mat转Sophus::SE3
        auto tmp_c = Converter::toSE3Quat(cur_frame_->mTcw);
        SE3f cur_Tcw_SE3 = SE3d(tmp_c.rotation(), tmp_c.translation()).cast<float>();
        auto tmp_r = Converter::toSE3Quat(ref_frame_->mTcw);
        SE3f ref_Tcw_SE3 = SE3d(tmp_r.rotation(), tmp_r.translation()).cast<float>();

//        SE3f T_cur_from_ref(cur_frame->mTcw * ref_frame_->mTcw.inverse());
//        cur_frame->mTcw * ref_frame_->mTcw.inverse() = Tcl
        SE3f T_cur_from_ref(cur_Tcw_SE3 * ref_Tcw_SE3.inverse());
cout<<"--------------------------------------------"<<endl;

        int iterations[] = {10, 10, 10, 10, 10, 10};
        //金字塔迭代，从最高层（即分辨率最低）开始迭代，到最低层（原始图像）
        float _chi2;
        for (level_ = max_level_; level_ >= min_level_; level_ -= 1) {
            mu_ = 0.1;
            pt_cache_.jacobian.setZero();
            seg_cache_.jacobian.setZero();
            edge_cache_.jacobian.setZero();
            lkSeg_cache_.jacobian.setZero();

            seg_cache_.visible_fts.resize(seg_cache_.ref_patch.rows,false);
            edge_cache_.visible_fts.resize(edge_cache_.ref_patch.rows,false);
            lkSeg_cache_.visible_fts.resize(lkSeg_cache_.ref_patch.rows,false);

            jacobian_cache_.setZero();
            have_ref_patch_cache_ = false;
//            n_iter_ = iterations[level_];
            n_iter_ = 10;
            segNum=0;
            edgeNum=0;
            lkSegNum=0;

            seg_cache_.seg_ref_patch.clear();
            edge_cache_.seg_ref_patch.clear();
            lkSeg_cache_.seg_ref_patch.clear();

            optimize(T_cur_from_ref,_chi2);
//            cout<<endl;
            if (_chi2 > sparseChi2 || n_meas_ < sparseNMeans){
                cout<<"_chi2:" <<_chi2<<" > x || n_meas_ < y:"<<n_meas_<<endl;
                return false;
            }

        }

        TCR = T_cur_from_ref;
        return n_meas_ / patch_area_;
    }

    Matrix<float, 6, 6> SparseImgAlign::getFisherInformation() {
        float sigma_i_sq = 5e-4 * 255 * 255; // image noise
        Matrix<float, 6, 6> I = H_ / sigma_i_sq;
        return I;
    }

    void SparseImgAlign::precomputeReferencePatches() {
        const int border = patch_halfsize_ + 1;                             //边界
        const cv::Mat &ref_img = ref_frame_->mvImagePyramid_zzw[level_];        //参考帧图像
        const int stride = ref_img.cols;                                    //步长
        const float scale = ref_frame_->mvInvScaleFactors[level_];          //金字塔层尺度
        const float focal_length = ref_frame_->fx; // 这里用fx或fy差别不大

        size_t feature_counter = 0;

        for (int i = 0; i < ref_frame_->N; i++, ++feature_counter) {
            MapPoint *mp = ref_frame_->mvpMapPoints[i];
            if (mp == nullptr || mp->isBad() || ref_frame_->mvbOutlier[i] == true)
                continue;

            // check if reference with patch size is within image
            //在当前金字塔层下的像素坐标值
            const cv::KeyPoint &kp = ref_frame_->mvKeys[i];
            const float u_ref = kp.pt.x * scale;
            const float v_ref = kp.pt.y * scale;
            const int u_ref_i = floorf(u_ref);
            const int v_ref_i = floorf(v_ref);
            if (u_ref_i - border < 0 || v_ref_i - border < 0 || u_ref_i + border >= ref_img.cols ||
                v_ref_i + border >= ref_img.rows)
                continue;

            //该特征是否可视在这里改变状态，初始化时为false
            visible_fts_[i] = true;

            // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
            // const double depth ( ( ( *it )->_mappoint->_pos_world - ref_pos ).norm() );
            // LOG(INFO)<<"depth = "<<depth<<", features depth = "<<(*it)->_depth<<endl;
            auto tmp = Converter::toSE3Quat(ref_frame_->mTcw);
            SE3f ref_Tcw_SE3 = SE3d(tmp.rotation(), tmp.translation()).cast<float>();
            Vector3f WorldPos;
            cv::cv2eigen(mp->GetWorldPos(),WorldPos);
            const Vector3f xyz_ref = ref_Tcw_SE3 * WorldPos;
//            const Vector3f xyz_ref = ref_frame_->mTcw * mp->GetWorldPos();

            // evaluate projection jacobian
            //获取2*6的投影雅克比矩阵，该雅克比没有乘以焦距，如下所示
            Matrix<float, 2, 6> frame_jac;
            frame_jac = JacobXYZ2Cam(xyz_ref);

            // compute bilateral interpolation weights for reference image
            //双线性插值参数
            const float subpix_u_ref = u_ref - u_ref_i;
            const float subpix_v_ref = v_ref - v_ref_i;
            const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
            const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;

            //cache_ptr：指向ref_patch_cache_的指针，前面仅开辟了内存空间，这里通过指针填值
            size_t pixel_counter = 0;
            float *cache_ptr = reinterpret_cast<float *> ( ref_patch_cache_.data ) + patch_area_ * feature_counter;
            for (int y = 0; y < patch_size_; ++y) {
                //指向参考帧像素的指针，4*4patch的左上角开始
                uint8_t *ref_img_ptr = (uint8_t *) ref_img.data + (v_ref_i + y - patch_halfsize_) * stride +
                                       (u_ref_i - patch_halfsize_);
                for (int x = 0; x < patch_size_; ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter) {
                    // precompute interpolated reference patch color
                    // 通过双线性插值，给ref_patch_cache_填值
                    *cache_ptr =
                            w_ref_tl * ref_img_ptr[0] + w_ref_tr * ref_img_ptr[1] + w_ref_bl * ref_img_ptr[stride] +
                            w_ref_br * ref_img_ptr[stride + 1];

                    // we use the inverse compositional: thereby we can take the gradient always at the same position
                    // get gradient of warped image (~gradient at warped position)
                    //计算像素梯度值，0.5*(u[1]-u[-1]), 0.5*(v[1]-v[-1])，其中的每个像素值都使用双线性插值获得
                    float dx = 0.5f * ((w_ref_tl * ref_img_ptr[1] + w_ref_tr * ref_img_ptr[2] +
                                        w_ref_bl * ref_img_ptr[stride + 1] + w_ref_br * ref_img_ptr[stride + 2])
                                       - (w_ref_tl * ref_img_ptr[-1] + w_ref_tr * ref_img_ptr[0] +
                                          w_ref_bl * ref_img_ptr[stride - 1] + w_ref_br * ref_img_ptr[stride]));
                    float dy = 0.5f * ((w_ref_tl * ref_img_ptr[stride] + w_ref_tr * ref_img_ptr[1 + stride] +
                                        w_ref_bl * ref_img_ptr[stride * 2] + w_ref_br * ref_img_ptr[stride * 2 + 1])
                                       - (w_ref_tl * ref_img_ptr[-stride] + w_ref_tr * ref_img_ptr[1 - stride] +
                                          w_ref_bl * ref_img_ptr[0] + w_ref_br * ref_img_ptr[1]));

                    // cache the jacobian
                    //计算像素雅克比，即像素梯度*投影雅克比
                    jacobian_cache_.col(feature_counter * patch_area_ + pixel_counter) =
                            (dx * frame_jac.row(0) + dy * frame_jac.row(1)) * (focal_length * scale);
                }
            }
        }
        have_ref_patch_cache_ = true;

    }


    float SparseImgAlign::computeResiduals(
            const SE3f &T_cur_from_ref,
            bool linearize_system,
            bool compute_weight_scale) {
        // Warp the (cur)rent image such that it aligns with the (ref)erence image
        //当前迭代金字塔层的图像
        const cv::Mat &cur_img = cur_frame_->mvImagePyramid_zzw[level_];

        //可忽略
        if (linearize_system && display_)
            resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));


        //预计算参考帧特征patch的缓存，即将ref_patch_cache_开辟的存储空间填上相应的值
        //可以暂时认为ref_patch_cache_中已经有值了
        if (have_ref_patch_cache_ == false)
            precomputeReferencePatches();

//        cout<<"jacobian_cache_ computeResiduals"<<endl<<"----------------------------"<<endl;
//        for (int i = 0; i < 10; ++i) {
//            cout<<jacobian_cache_.col(i)<<endl<<"----------------------------"<<endl;
//        }
//        getchar();

        // compute the weights on the first iteration
        //可忽略
        std::vector<float> errors;
        if (compute_weight_scale)
            errors.reserve(visible_fts_.size());


        const int stride = cur_img.cols;
        const int border = patch_halfsize_ + 1;                         //patch的边界
        const float scale = ref_frame_->mvInvScaleFactors[level_];      //对应金字塔层
        float chi2 = 0.0;                                               //光度误差cost
        size_t feature_counter = 0; // is used to compute the index of the cached jacobian

        size_t visible = 0;

        for (int i = 0; i < ref_frame_->N; i++, feature_counter++) {
            // check if feature is within image
            if (visible_fts_[i] == false)
                continue;
            MapPoint *mp = ref_frame_->mvpMapPoints[i];
            assert(mp != nullptr);

            // compute pixel location in cur img
            auto tmp = Converter::toSE3Quat(ref_frame_->mTcw);
            SE3f ref_Tcw_SE3 = SE3d(tmp.rotation(), tmp.translation()).cast<float>();
            Vector3f WorldPos;
            cv::cv2eigen(mp->GetWorldPos(),WorldPos);
            const Vector3f xyz_ref = ref_Tcw_SE3 * WorldPos;
//            const Vector3f xyz_ref = ref_frame_->mTcw * mp->GetWorldPos();
            const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);

            const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
            const Vector2f uv_cur_pyr(uv_cur * scale);
            const float u_cur = uv_cur_pyr[0];
            const float v_cur = uv_cur_pyr[1];
            //floorf向下取整
            const int u_cur_i = floorf(u_cur);
            const int v_cur_i = floorf(v_cur);

            // check if projection is within the image
            if (u_cur_i < 0 || v_cur_i < 0 || u_cur_i - border < 0 || v_cur_i - border < 0 ||
                u_cur_i + border >= cur_img.cols || v_cur_i + border >= cur_img.rows)
                continue;

            visible++;

            // compute bilateral interpolation weights for the current image
            //通过双线性插值计算像素光度值
            const float subpix_u_cur = u_cur - u_cur_i;
            const float subpix_v_cur = v_cur - v_cur_i;
            //双线性插值参数，tl：topleft，tr：topright，bl：bottomleft，br：bottomright
            const float w_cur_tl = (1.0 - subpix_u_cur) * (1.0 - subpix_v_cur);
            const float w_cur_tr = subpix_u_cur * (1.0 - subpix_v_cur);
            const float w_cur_bl = (1.0 - subpix_u_cur) * subpix_v_cur;
            const float w_cur_br = subpix_u_cur * subpix_v_cur;

            //指向参考帧特征patch的指针：头指针+特征数*每个特征patch的像素个数
            float *ref_patch_cache_ptr = reinterpret_cast<float *> ( ref_patch_cache_.data ) + patch_area_ * feature_counter;
            size_t pixel_counter = 0; // is used to compute the index of the cached jacobian
            for (int y = 0; y < patch_size_; ++y) {
                //指向当前帧像素值的指针，4*4patch的左上角开始
                uint8_t *cur_img_ptr = (uint8_t *) cur_img.data + (v_cur_i + y - patch_halfsize_) * stride +
                                       (u_cur_i - patch_halfsize_);

                for (int x = 0; x < patch_size_; ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr) {
                    // compute residual
                    // 根据双线性插值计算当前pixel的像素值
                    const float intensity_cur =
                            w_cur_tl * cur_img_ptr[0] + w_cur_tr * cur_img_ptr[1] + w_cur_bl * cur_img_ptr[stride] +
                            w_cur_br * cur_img_ptr[stride + 1];
                    //计算残差：当前帧-参考帧
                    const float res = intensity_cur - (*ref_patch_cache_ptr);

                    // used to compute scale for robust cost
                    // 可忽略
                    if (compute_weight_scale)
                        errors.push_back(fabsf(res));

                    // robustification
                    // 可忽略
                    float weight = 1.0;
                    if (use_weights_) {
                        weight = weight_function_->value(res / scale_);
                    }

                    //差值平方累加和
                    chi2 += res * res * weight;
                    n_meas_++;

//                    if(i<10){
//                        cout<<endl<<"======================"<<endl;
//                        cout<<"i: "<<i<<"  x: "<<x<<"  y: "<<y<<endl;
//                        cout<<"intensity_cur "<<intensity_cur<<"  ref_patch_cache_ptr "<<*ref_patch_cache_ptr<<endl;
//                        cout<<"res "<<res<<endl;
//                        cout<<"chi2 "<<chi2<<endl;
//                        cout<<endl<<"======================"<<endl;
//                    }


                    //求解雅克比过程
                    if (linearize_system) {
                        // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                        // 取出当前特征对应的雅克比矩阵，因为使用的是逆向组合算法，所以jacobian_cache_预先计算好了，存储了所有特征的雅克比
                        const Sophus::Vector6f J(jacobian_cache_.col(feature_counter * patch_area_ + pixel_counter));
                        H_.noalias() += J * J.transpose() * weight;
                        Jres_.noalias() -= J * res * weight;
                        if (display_)
                            resimg_.at<float>((int) v_cur + y - patch_halfsize_, (int) u_cur + x - patch_halfsize_) =
                                    res / 255.0;
                    }
                }
            }
        }


        // compute the weights on the first iteration
        if (compute_weight_scale && iter_ == 0)
            scale_ = scale_estimator_->compute(errors);
        return chi2 / n_meas_;
    }

    float SparseImgAlign::computeResiduals_zzw(
            const SE3f &T_cur_from_ref,
            bool linearize_system,
            bool compute_weight_scale) {
        // Warp the (cur)rent image such that it aligns with the (ref)erence image
        //当前迭代金字塔层的图像
        const cv::Mat &cur_img = cur_frame_->mvImagePyramid_zzw[level_];

        //可忽略
        //display_ default false
        if (linearize_system && display_)
            resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));

        //预计算参考帧特征patch的缓存，即将ref_patch_cache_开辟的存储空间填上相应的值
        //可以暂时认为ref_patch_cache_中已经有值了
        if (have_ref_patch_cache_ == false)
            precomputeReferencePatches_zzw();


        // compute the weights on the first iteration
        //可忽略
        std::vector<float> errors;
        if (compute_weight_scale)
            errors.reserve(visible_fts_.size());

        Matrix<float, 6, 6> pt_H;
        Matrix<float, 6, 1> pt_Jres;
        // define other interest variables
        std::vector<float> pt_errors;
        float pt_chi2 = 0.0;
        computeGaussNewtonParamsPoints(
                T_cur_from_ref, linearize_system, compute_weight_scale,
                pt_cache_, pt_H, pt_Jres, pt_errors, pt_chi2 );

        Matrix<float, 6, 6> seg_H;
        Matrix<float, 6, 1> seg_Jres;
        // define other interest variables
        std::vector<float> seg_errors;
        float seg_chi2 = 0.0;
        // compute the parameters for Gauss-Newton update
        computeGaussNewtonParamsSegments(
                T_cur_from_ref, linearize_system, compute_weight_scale,
                seg_cache_, seg_H, seg_Jres, seg_errors, seg_chi2 );

        Matrix<float, 6, 6> edge_H;
        Matrix<float, 6, 1> edge_Jres;
        std::vector<float> edge_errors;
        float edge_chi2 = 0.0;
        computeGaussNewtonParamsPlaneEdge(
                T_cur_from_ref, linearize_system, compute_weight_scale,
                edge_cache_, edge_H, edge_Jres, edge_errors, edge_chi2 );

        Matrix<float, 6, 6> lkSeg_H;
        Matrix<float, 6, 1> lkSeg_Jres;
        std::vector<float> lkSeg_errors;
        float lkSeg_chi2 = 0.0;
        computeGaussNewtonParamsLKSegments(
                T_cur_from_ref, linearize_system, compute_weight_scale,
                lkSeg_cache_, lkSeg_H, lkSeg_Jres, lkSeg_errors, lkSeg_chi2 );

        if(linearize_system)
        {
            // sum the contribution from both points and segments
//            H_    = pt_H    + seg_H    + edge_H;
//            Jres_ = pt_Jres + seg_Jres + edge_Jres;
//            H_    = pt_H    + seg_H;
//            Jres_ = pt_Jres + seg_Jres;
//            H_    = pt_H    + seg_H + lkSeg_H;
//            Jres_ = pt_Jres + seg_Jres + lkSeg_Jres;
            H_    = pt_H    + seg_H + lkSeg_H + edge_H;
            Jres_ = pt_Jres + seg_Jres + lkSeg_Jres + edge_Jres;
        }

        float chi2 = pt_chi2 + seg_chi2 + lkSeg_chi2 + edge_chi2;
//        float chi2 = pt_chi2 + seg_chi2;
//cout<<"pt_chi2 "<<pt_chi2<<"  ptNum "<<(n_meas_-segNum-lkSegNum)<<endl;
//cout<<"seg_chi2 "<<seg_chi2<<"  segNum "<<segNum<<endl;
//cout<<"lkSeg_chi2 "<<lkSeg_chi2<<"  lkSegNum "<<lkSegNum<<endl;
//cout<<"chi2/n_meas_ "<<chi2 / n_meas_<<endl<<endl;

//if(seg_chi2==0){
//    cout<<"  NL "<<ref_frame_->NL;
//    getchar();
//}



        // compute the weights on the first iteration
        if (compute_weight_scale && iter_ == 0)
            scale_ = scale_estimator_->compute(errors);
        return chi2 / n_meas_;
    }

    void SparseImgAlign::computeGaussNewtonParamsPoints(
            const SE3f &T_cur_from_ref,
            bool linearize_system,
            bool compute_weight_scale,
            Cache& cache,
            Matrix<float, 6, 6> &H,
            Matrix<float, 6, 1> &Jres,
            std::vector<float>& errors,
            float& chi2)
    {
        Patch patch( patch_size_, cur_frame_->mvImagePyramid_zzw[level_] );
        Patch resPatch;
        //display_ default false
        if(linearize_system && display_)
            resPatch = Patch( patch_size_, resimg_ );

        // compute the weights on the first iteration
        if(compute_weight_scale)
            errors.reserve(cache.visible_fts.size());


        const float scale = ref_frame_->mvInvScaleFactors[level_];

        // reset chi2 variable to zero
        chi2 = 0.0;
        H.setZero();
        Jres.setZero();
        size_t feature_counter = 0;
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

        for(int i = 0; i < ref_frame_->N; i++, ++feature_counter, ++visiblity_it)
        {

            if(!*visiblity_it)
                continue;

            MapPoint *mp = ref_frame_->mvpMapPoints[i];
            assert(mp != nullptr);

            float depth = ref_frame_->mvDepth[i];
            if (depth < 0)
                continue;

            const cv::KeyPoint &kp = ref_frame_->mvKeys[i];
//            float depth = ref_frame_->mvDepth[i];
            Vector3f xyz_ref;
            xyz_ref[0] = (kp.pt.x - ref_frame_->cx) * depth * ref_frame_->invfx;
            xyz_ref[1] = (kp.pt.y - ref_frame_->cy) * depth * ref_frame_->invfy;
            xyz_ref[2] = depth;


            const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);
            const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
            const Vector2f uv_cur_pyr(uv_cur * scale);

            patch.setPosition(uv_cur_pyr);
            if(!patch.isInFrame(patch.halfsize+1))
                continue;

            patch.computeInterpWeights();
            patch.setRoi();


            size_t pixel_counter = 0;
            float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + patch.area*feature_counter;
            uint8_t* img_ptr;
            const int stride = patch.stride;
            cv::MatIterator_<float> itDisp;
            //display_ default false
            if(linearize_system && display_)
            {
                resPatch.setPosition(uv_cur_pyr);
                resPatch.setRoi();
                itDisp = resPatch.roi.begin<float>();
            }


            for(int y=0; y<patch.size; ++y)
            {
                img_ptr = patch.roi.ptr(y);
                for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++pixel_counter)
                {

                    const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                    const float res = intensity_cur - (*cache_ptr);
                    const float res2 = res*res;

                    if(compute_weight_scale)
                        errors.push_back(fabsf(res));

                    // robustification
                    float weight = 1.0;

                    chi2 += res*res*weight;
                    n_meas_++;

                    if(linearize_system)
                    {
                        // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                        const Sophus::Vector6f J(cache.jacobian.col(feature_counter*patch.area + pixel_counter));
                        H.noalias() += J*J.transpose()*weight;
                        Jres.noalias() -= J*res*weight;
                        //display_ default false
                        if(display_)
                        {
                            *itDisp = res/255.0;
                            ++itDisp;
                        }
                    }
                }
            }
        }
    }

    void SparseImgAlign::computeGaussNewtonParamsSegments(
            const SE3f &T_cur_from_ref,
            bool linearize_system,
            bool compute_weight_scale,
            Cache &cache,
            Matrix<float, 6, 6> &H,
            Matrix<float, 6, 1> &Jres,
            std::vector<float> &errors,
            float &chi2)
    {
        Patch patch( patch_size_, cur_frame_->mvImagePyramid_zzw[level_] );
        Patch resPatch;
        //display_ default false
        if(linearize_system && display_)
            resPatch = Patch( patch_size_, resimg_ );

        if(compute_weight_scale)
            errors.reserve(cache.visible_fts.size());

        const float scale = ref_frame_->mvInvScaleFactors[level_];

        // reset chi2 variable to zero
        chi2 = 0.0;

        // set GN parameters to zero prior to accumulate results
        H.setZero();
        Jres.setZero();
        std::vector<size_t>::iterator offset_it = patch_offset.begin();
        size_t cache_idx = 0;
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

        Matrix<float,6,6> H_ls    = Matrix<float, 6, 6>::Zero();
        Matrix<float,6,1> Jres_ls = Matrix<float, 6, 1>::Zero();

        for(int i = 0; i < ref_frame_->NL; i++, ++offset_it, ++visiblity_it)
        {
            if(!*visiblity_it)
                continue;

            std::pair<float, float> depth = ref_frame_->mvDepthLine_zzw[i];

            const cv::line_descriptor::KeyLine kl = ref_frame_->mvKeylinesUn[i];
            Vector2f sp_l(kl.startPointX,kl.startPointY),ep_l(kl.endPointX,kl.endPointY);
            float length = (ep_l-sp_l).norm();

            cache_idx = *offset_it;

            Vector2f inc2f;

            size_t N_samples = setupSampling(patch.size, inc2f, sp_l, ep_l, length);

            N_samples = 1 + (N_samples-1) * scale;

//            std::pair<float, float> depth = ref_frame_->mvDepthLine_zzw[i];
            Vector3f p_ref,q_ref;
            p_ref[0] = (sp_l[0] - ref_frame_->cx) * depth.first * ref_frame_->invfx;
            p_ref[1] = (sp_l[1] - ref_frame_->cy) * depth.first * ref_frame_->invfy;
            p_ref[2] = depth.first;
            q_ref[0] = (ep_l[0] - ref_frame_->cx) * depth.second * ref_frame_->invfx;
            q_ref[1] = (ep_l[1] - ref_frame_->cy) * depth.second * ref_frame_->invfy;
            q_ref[2] = depth.second;

            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;


            // Evaluate over the patch for each point sampled in the segment (including extremes)
            Matrix<float,6,6> H_    = Matrix<float, 6, 6>::Zero();
            Matrix<float,6,1> Jres_ = Matrix<float, 6, 1>::Zero();
            vector<float> ls_res;
            bool good_line = true;

            for(unsigned int sample = 0; sample < N_samples; ++sample, xyz_ref+=inc3f )
            {
                const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);
                const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
                const Vector2f uv_cur_pyr(uv_cur * scale);

                patch.setPosition(uv_cur_pyr);

                if(!patch.isInFrame(patch.halfsize+1))
                {
                    cache_idx += patch.size; // Do not lose position of the next patch in cache!
                    good_line = false;
                    sample    = N_samples;
                    continue;
                }

                patch.computeInterpWeights();

                patch.setRoi();

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;
                const int stride = patch.stride;
                cv::MatIterator_<float> itDisp;
                //display_ default false
                if(linearize_system && display_)
                {
                    resPatch.setPosition(uv_cur_pyr);
                    resPatch.setRoi();
                    itDisp = resPatch.roi.begin<float>();
                }

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

                        const float res = intensity_cur - (*cache_ptr);

                        ls_res.push_back(res);

                        if(linearize_system)
                        {
                            // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                            const Sophus::Vector6f J(cache.jacobian.col(cache_idx));
                            H_.noalias()     += J*J.transpose() ;
                            Jres_.noalias()  -= J*res;
                            //display_ default false
                            if(display_)
                            {
                                *itDisp = res/255.0;
                                ++itDisp;
                            }
                        }
                    }//end col-sweep of current row
                }//end row-sweep
            }//end segment-sweep

            float res_ = 0.0, res2_;
            for(vector<float>::iterator it = ls_res.begin(); it != ls_res.end(); ++it){
                res_ += fabsf(*it);
            }

            res_ = res_ / double(N_samples);

            if( good_line && res_ < 200.0)  // debug
            {
                // used to compute scale for robust cost
                if(compute_weight_scale)
                    errors.push_back(res_);
                // robustification
                float weight = 1.0;
                // update total H and J
                H.noalias()    += H_    * weight / res_;  // only divide hessian once (H/res2 g/res)
                Jres.noalias() += Jres_ * weight ;        // it is already negative
                chi2           += res_*res_*weight;
                n_meas_++;
                segNum++;

            }

            good_line = true;
            ls_res.clear();
            H_.setZero();
            Jres_.setZero();
        }//end feature-sweep
    }


    void SparseImgAlign::computeGaussNewtonParamsPlaneEdge(
            const SE3f &T_cur_from_ref,
            bool linearize_system,
            bool compute_weight_scale,
            Cache &cache,
            Matrix<float, 6, 6> &H,
            Matrix<float, 6, 1> &Jres,
            std::vector<float> &errors,
            float &chi2)
    {
        Patch patch( patch_size_, cur_frame_->mvImagePyramid_zzw[level_] );
        Patch resPatch;
        //display_ default false
        if(linearize_system && display_)
            resPatch = Patch( patch_size_, resimg_ );

        if(compute_weight_scale)
            errors.reserve(cache.visible_fts.size());

        const float scale = ref_frame_->mvInvScaleFactors[level_];

        // reset chi2 variable to zero
        chi2 = 0.0;

        // set GN parameters to zero prior to accumulate results
        H.setZero();
        Jres.setZero();
        std::vector<size_t>::iterator offset_it = patch_offset_edge.begin();
        size_t cache_idx = 0;
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

        const int NE = ref_frame_->allPlaneEdgeLine.size();

        Matrix<float,6,6> H_ls    = Matrix<float, 6, 6>::Zero();
        Matrix<float,6,1> Jres_ls = Matrix<float, 6, 1>::Zero();

        for(int i = 0; i < NE; i++, ++offset_it, ++visiblity_it)
        {
            if(!*visiblity_it)
                continue;

            Vector3f p_ref;
            Vector3f q_ref;
            p_ref[0] = ref_frame_->allPlaneEdgeLine[i].first.x;
            p_ref[1] = ref_frame_->allPlaneEdgeLine[i].first.y;
            p_ref[2] = ref_frame_->allPlaneEdgeLine[i].first.z;
            q_ref[0] = ref_frame_->allPlaneEdgeLine[i].second.x;
            q_ref[1] = ref_frame_->allPlaneEdgeLine[i].second.y;
            q_ref[2] = ref_frame_->allPlaneEdgeLine[i].second.z;

            if (p_ref[2] <= 0 || q_ref[2]<= 0)
                continue;

            Vector2f sp_edge = ref_frame_->Camera2Pixel(p_ref);
            Vector2f ep_edge = ref_frame_->Camera2Pixel(q_ref);
            float length = (ep_edge-sp_edge).norm();

            cache_idx = *offset_it;

            Vector2f inc2f;

            size_t N_samples = setupSampling(patch.size, inc2f, sp_edge, ep_edge, length);

            N_samples = 1 + (N_samples-1) * scale;

            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;


            // Evaluate over the patch for each point sampled in the segment (including extremes)
            Matrix<float,6,6> H_    = Matrix<float, 6, 6>::Zero();
            Matrix<float,6,1> Jres_ = Matrix<float, 6, 1>::Zero();
            vector<float> ls_res;
            bool good_line = true;

            for(unsigned int sample = 0; sample < N_samples; ++sample, xyz_ref+=inc3f )
            {
                const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);
                const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
                const Vector2f uv_cur_pyr(uv_cur * scale);

                if( cur_frame_->mnId>28000 ) {
                    cv::Mat img, img1;
                    patch.full_img.copyTo(img);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
                    cv::Point2f curP(uv_cur_pyr[0], uv_cur_pyr[1]);
                    cv::Point2f curPEnd(cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)[0]* scale, cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)[1]* scale);

                    cv::Point2f preP(preCompute[i][sample][0], preCompute[i][sample][1]);
                    cv::Point2f prePEnd(preComputeEnd[i][sample][0], preComputeEnd[i][sample][1]);


                    cout << " curP:" << curP << " preP:" << preP<< endl;
                    cout <<"xyz_cur ["<<xyz_cur[0]<<" "<<xyz_cur[1]<<" "<<xyz_cur[2]<<"]"<< endl;
                    cout <<"uv_cur ["<<uv_cur[0]<<" "<<uv_cur[1]<<"]"<< endl;


                    int curx = floorf(curP.x), cury = floorf(curP.y);
                    float curPix = *(patch.full_img.data + cury * patch.stride + curx);
                    int prex = floorf(preP.x), prey = floorf(preP.y);
                    float prePix = *(ref_frame_->mvImagePyramid_zzw[level_].data + prey * patch.stride + prex);

//                    cv::line(img, curP, curPEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
//                    cv::line(img1, preP, prePEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
//
//
//                    cv::imshow("cur", img);
//                    cv::imshow("pre", img1);
                    cout << "intensity:" << curPix - prePix << endl;
//                    getchar();
                }

                patch.setPosition(uv_cur_pyr);

                if(!patch.isInFrame(patch.halfsize+1))
                {
                    cache_idx += patch.size; // Do not lose position of the next patch in cache!
                    good_line = false;
                    sample    = N_samples;
                    continue;
                }

                patch.computeInterpWeights();

                patch.setRoi();

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;
                const int stride = patch.stride;
                cv::MatIterator_<float> itDisp;
                //display_ default false
                if(linearize_system && display_)
                {
                    resPatch.setPosition(uv_cur_pyr);
                    resPatch.setRoi();
                    itDisp = resPatch.roi.begin<float>();
                }

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

                        const float res = intensity_cur - (*cache_ptr);
                        ls_res.push_back(res);

                        if(linearize_system)
                        {
                            // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                            const Sophus::Vector6f J(cache.jacobian.col(cache_idx));
                            H_.noalias()     += J*J.transpose() ;
                            Jres_.noalias()  -= J*res;
                            //display_ default false
                            if(display_)
                            {
                                *itDisp = res/255.0;
                                ++itDisp;
                            }
                        }
                    }//end col-sweep of current row
                }//end row-sweep
            }//end segment-sweep

            float res_ = 0.0, res2_;
            for(vector<float>::iterator it = ls_res.begin(); it != ls_res.end(); ++it){
                res_ += fabsf(*it);
            }

            res_ = res_ / double(N_samples);

            if( good_line && res_ < 200.0)  // debug
            {
                // used to compute scale for robust cost
                if(compute_weight_scale)
                    errors.push_back(res_);
                // robustification
                float weight = 1.0;
                // update total H and J
                H.noalias()    += H_    * weight / res_;  // only divide hessian once (H/res2 g/res)
                Jres.noalias() += Jres_ * weight ;        // it is already negative
                chi2           += res_*res_*weight;
                n_meas_++;
                edgeNum++;

            }

            good_line = true;
            ls_res.clear();
            H_.setZero();
            Jres_.setZero();
        }//end feature-sweep
    }

    void SparseImgAlign::computeGaussNewtonParamsLKSegments(
            const SE3f &T_cur_from_ref,
            bool linearize_system,
            bool compute_weight_scale,
            Cache &cache,
            Matrix<float, 6, 6> &H,
            Matrix<float, 6, 1> &Jres,
            std::vector<float> &errors,
            float &chi2)
    {
        Patch patch( patch_size_, cur_frame_->mvImagePyramid_zzw[level_] );
        Patch resPatch;
        //display_ default false
        if(linearize_system && display_)
            resPatch = Patch( patch_size_, resimg_ );

        if(compute_weight_scale)
            errors.reserve(cache.visible_fts.size());

        const float scale = ref_frame_->mvInvScaleFactors[level_];

        // reset chi2 variable to zero
        chi2 = 0.0;

        // set GN parameters to zero prior to accumulate results
        H.setZero();
        Jres.setZero();
        std::vector<size_t>::iterator offset_it = patch_offset_lkSeg.begin();
        size_t cache_idx = 0;
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

        Matrix<float,6,6> H_ls    = Matrix<float, 6, 6>::Zero();
        Matrix<float,6,1> Jres_ls = Matrix<float, 6, 1>::Zero();

        int NLK = ref_frame_->mLKLine.size();

        for(int i = 0; i < NLK; i++, ++offset_it, ++visiblity_it)
        {
            if(!*visiblity_it)
                continue;

            std::pair<float, float> depth;
            depth = make_pair(ref_frame_->mLKLine[i].first[2],ref_frame_->mLKLine[i].second[2]);

            Vector2f sp_l,ep_l;
            float length;
            sp_l<<ref_frame_->mLKLine[i].first[0],ref_frame_->mLKLine[i].first[1];
            ep_l<<ref_frame_->mLKLine[i].second[0],ref_frame_->mLKLine[i].second[1];
            length = (ep_l-sp_l).norm();

            cache_idx = *offset_it;

            Vector2f inc2f;
            size_t N_samples = setupSampling(patch.size, inc2f, sp_l, ep_l, length);
            N_samples = 1 + (N_samples-1) * scale;

            Vector3f p_ref,q_ref;
            p_ref[0] = (sp_l[0] - ref_frame_->cx) * depth.first * ref_frame_->invfx;
            p_ref[1] = (sp_l[1] - ref_frame_->cy) * depth.first * ref_frame_->invfy;
            p_ref[2] = depth.first;
            q_ref[0] = (ep_l[0] - ref_frame_->cx) * depth.second * ref_frame_->invfx;
            q_ref[1] = (ep_l[1] - ref_frame_->cy) * depth.second * ref_frame_->invfy;
            q_ref[2] = depth.second;

            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;


            // Evaluate over the patch for each point sampled in the segment (including extremes)
            Matrix<float,6,6> H_    = Matrix<float, 6, 6>::Zero();
            Matrix<float,6,1> Jres_ = Matrix<float, 6, 1>::Zero();
            vector<float> ls_res;
            bool good_line = true;

            for(unsigned int sample = 0; sample < N_samples; ++sample, xyz_ref+=inc3f )
            {
                const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);
                const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
                const Vector2f uv_cur_pyr(uv_cur * scale);

                patch.setPosition(uv_cur_pyr);

                if(!patch.isInFrame(patch.halfsize+1))
                {
                    cache_idx += patch.size; // Do not lose position of the next patch in cache!
                    good_line = false;
                    sample    = N_samples;
                    continue;
                }

                patch.computeInterpWeights();

                patch.setRoi();

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;
                const int stride = patch.stride;
                cv::MatIterator_<float> itDisp;
                //display_ default false
                if(linearize_system && display_)
                {
                    resPatch.setPosition(uv_cur_pyr);
                    resPatch.setRoi();
                    itDisp = resPatch.roi.begin<float>();
                }

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

                        const float res = intensity_cur - (*cache_ptr);

                        ls_res.push_back(res);

                        if(linearize_system)
                        {
                            // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                            const Sophus::Vector6f J(cache.jacobian.col(cache_idx));
                            H_.noalias()     += J*J.transpose() ;
                            Jres_.noalias()  -= J*res;
                            //display_ default false
                            if(display_)
                            {
                                *itDisp = res/255.0;
                                ++itDisp;
                            }
                        }
                    }//end col-sweep of current row
                }//end row-sweep
            }//end segment-sweep

            float res_ = 0.0, res2_;
            for(vector<float>::iterator it = ls_res.begin(); it != ls_res.end(); ++it){
                res_ += fabsf(*it);
            }

            res_ = res_ / double(N_samples);

            if( good_line && res_ < 200.0)  // debug
            {
                // used to compute scale for robust cost
                if(compute_weight_scale)
                    errors.push_back(res_);
                // robustification
                float weight = 1.0;
                // update total H and J
                H.noalias()    += H_    * weight / res_;  // only divide hessian once (H/res2 g/res)
                Jres.noalias() += Jres_ * weight ;        // it is already negative
                chi2           += res_*res_*weight;
                n_meas_++;
                lkSegNum++;

            }

            good_line = true;
            ls_res.clear();
            H_.setZero();
            Jres_.setZero();
        }//end feature-sweep
    }

    void SparseImgAlign::precomputeReferencePatches_zzw()
    {
        precomputeGaussNewtonParamsPoints(pt_cache_);
        precomputeGaussNewtonParamsSegments(seg_cache_);
        precomputeGaussNewtonParamsPlaneEdge(edge_cache_);
        precomputeGaussNewtonParamsLKSegments(lkSeg_cache_);

        // set flag to true to avoid repeating unnecessary computations in the following iterations
        have_ref_patch_cache_ = true;
    }

    void SparseImgAlign::precomputeGaussNewtonParamsPoints(Cache &cache)
    {
        Patch patch( patch_size_, ref_frame_->mvImagePyramid_zzw[level_] );

        const float scale = ref_frame_->mvInvScaleFactors[level_];
        const float focal_length = ref_frame_->fx;

        {
            size_t feature_counter = 0;
            std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

            for(int i = 0; i < ref_frame_->N; i++, ++feature_counter, ++visiblity_it)
            {

                MapPoint *mp = ref_frame_->mvpMapPoints[i];
                if (mp == nullptr || mp->isBad() || ref_frame_->mvbOutlier[i] == true)
                    continue;

                float depth = ref_frame_->mvDepth[i];
                if (depth < 0)
                    continue;

                const cv::KeyPoint &kp = ref_frame_->mvKeys[i];
                Vector2f refPoint(kp.pt.x,kp.pt.y);

                patch.setPosition(refPoint*scale);

                if(!patch.isInFrame(patch.halfsize+1))
                    continue;
                patch.computeInterpWeights();
                patch.setRoi();

                // flag the feature as valid/visible
                *visiblity_it = true;

                Vector3f xyz_ref;
                xyz_ref[0] = (kp.pt.x - ref_frame_->cx) * depth * ref_frame_->invfx;
                xyz_ref[1] = (kp.pt.y - ref_frame_->cy) * depth * ref_frame_->invfy;
                xyz_ref[2] = depth;


                Matrix<float,2,6> frame_jac;
                frame_jac = JacobXYZ2Cam(xyz_ref);


                size_t pixel_counter = 0;
                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + patch.area*feature_counter;
                uint8_t* img_ptr;
                const int stride = patch.stride;

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++pixel_counter)
                    {
                        *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

                        float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
                                           -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
                        float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
                                           -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));

                        pt_cache_.jacobian.col(feature_counter*patch.area + pixel_counter) =
                                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length * scale);

                    }
                }
            }
        }
    }

    void SparseImgAlign::precomputeGaussNewtonParamsSegments(Cache &cache)
    {
        // initialize patch parameters (mainly define its geometry)
        Patch patch( patch_size_, ref_frame_->mvImagePyramid_zzw[level_] );

        const float scale = ref_frame_->mvInvScaleFactors[level_];
        const float focal_length = ref_frame_->fx;
        const int border = patch_halfsize_ + 1;

        // TODO: feature_counter is no longer valid because each segment
        //  has a variable number of patches (and total pixels)
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();
        patch_offset = std::vector<size_t>(ref_frame_->NL,0);
        std::vector<size_t>::iterator offset_it = patch_offset.begin();
        size_t cache_idx = 0;

        for(int i = 0; i < ref_frame_->NL; i++, ++visiblity_it, ++offset_it)
        {
            *offset_it = cache_idx;

//            MapLine *ml = ref_frame_->mvpMapLines[i];
//            if (!ml || ml->isBad() || ref_frame_->mvbLineOutlier[i]) {
//                continue;
//            }

            const cv::line_descriptor::KeyLine kl = ref_frame_->mvKeylinesUn[i];
            Vector2f sp_l(kl.startPointX,kl.startPointY),ep_l(kl.endPointX,kl.endPointY);
            float length = (ep_l-sp_l).norm();

            const float u_ref_s = kl.startPointX*scale;
            const float v_ref_s = kl.startPointY*scale;
            const float u_ref_e = kl.endPointX*scale;
            const float v_ref_e = kl.endPointY*scale;
            const int u_ref_i_s = floorf(u_ref_s);
            const int v_ref_i_s = floorf(v_ref_s);
            const int u_ref_i_e = floorf(u_ref_e);
            const int v_ref_i_e = floorf(v_ref_e);

            if (u_ref_i_s - border < 0 || v_ref_i_s - border < 0 || u_ref_i_s + border >= patch.full_img.cols ||
                v_ref_i_s + border >= patch.full_img.rows)
                continue;
            if (u_ref_i_e - border < 0 || v_ref_i_e - border < 0 || u_ref_i_e + border >= patch.full_img.cols ||
                v_ref_i_e + border >= patch.full_img.rows)
                continue;

            std::pair<float, float> depth = ref_frame_->mvDepthLine_zzw[i];
            if (depth.first <= 0 || depth.second <= 0)
                continue;

            *visiblity_it = true;


            Vector2f inc2f;

            size_t N_samples = setupSampling(patch.size, inc2f, sp_l, ep_l, length);

            N_samples = 1 + (N_samples-1) * scale;

            // Parameterize 2D segment
            inc2f = inc2f * scale / (N_samples-1);
            Vector2f px_ref = sp_l * scale;


            Vector3f p_ref,q_ref;
            p_ref[0] = (sp_l[0] - ref_frame_->cx) * depth.first * ref_frame_->invfx;
            p_ref[1] = (sp_l[1] - ref_frame_->cy) * depth.first * ref_frame_->invfy;
            p_ref[2] = depth.first;
            q_ref[0] = (ep_l[0] - ref_frame_->cx) * depth.second * ref_frame_->invfx;
            q_ref[1] = (ep_l[1] - ref_frame_->cy) * depth.second * ref_frame_->invfy;
            q_ref[2] = depth.second;


            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;

            for(unsigned int sample = 0; sample<N_samples; ++sample, px_ref+=inc2f, xyz_ref+=inc3f )
            {
                patch.setPosition( px_ref );
                patch.computeInterpWeights();
                patch.setRoi();

                // evaluate projection jacobian
                Matrix<float,2,6> frame_jac;
                frame_jac = JacobXYZ2Cam(xyz_ref);

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;
                const int stride = patch.stride;
                std::vector<float> cache_ptr_seg;

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                        cache_ptr_seg.push_back(*cache_ptr);

                        float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
                                           -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
                        float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
                                           -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));

                        // cache the jacobian
                        seg_cache_.jacobian.col(cache_idx) =
                                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length * scale);

                    }//end col-sweep in current row
                }//end row-sweep
                cache.seg_ref_patch[i][sample] = cache_ptr_seg;
            }//end segment-sweep (through sampled patches)
        }//end feature-sweep
    }

    void SparseImgAlign::precomputeGaussNewtonParamsPlaneEdge(Cache &cache)
    {
        // initialize patch parameters (mainly define its geometry)
        Patch patch( patch_size_, ref_frame_->mvImagePyramid_zzw[level_] );

        const float scale = ref_frame_->mvInvScaleFactors[level_];
        const float focal_length = ref_frame_->fx;
        const int border = patch_halfsize_ + 1;

        const int NE = ref_frame_->allPlaneEdgeLine.size();

        // TODO: feature_counter is no longer valid because each segment
        //  has a variable number of patches (and total pixels)
        auto visiblity_it = cache.visible_fts.begin();
        patch_offset_edge = std::vector<size_t>(NE,0);
        auto offset_it = patch_offset_edge.begin();
        size_t cache_idx = 0;



        for(int i = 0; i < NE; i++, ++visiblity_it, ++offset_it)
        {
            *offset_it = cache_idx;


            Vector3f p_ref;
            Vector3f q_ref;
            p_ref[0] = ref_frame_->allPlaneEdgeLine[i].first.x;
            p_ref[1] = ref_frame_->allPlaneEdgeLine[i].first.y;
            p_ref[2] = ref_frame_->allPlaneEdgeLine[i].first.z;
            q_ref[0] = ref_frame_->allPlaneEdgeLine[i].second.x;
            q_ref[1] = ref_frame_->allPlaneEdgeLine[i].second.y;
            q_ref[2] = ref_frame_->allPlaneEdgeLine[i].second.z;

            if (p_ref[2] <= 0 || q_ref[2]<= 0)
                continue;

            Vector2f sp_edge = ref_frame_->Camera2Pixel(p_ref);
            Vector2f ep_edge = ref_frame_->Camera2Pixel(q_ref);

            float length = (ep_edge-sp_edge).norm();

            const float u_ref_s = sp_edge[0]*scale;
            const float v_ref_s = sp_edge[1]*scale;
            const float u_ref_e = ep_edge[0]*scale;
            const float v_ref_e = ep_edge[1]*scale;
            const int u_ref_i_s = floorf(u_ref_s);
            const int v_ref_i_s = floorf(v_ref_s);
            const int u_ref_i_e = floorf(u_ref_e);
            const int v_ref_i_e = floorf(v_ref_e);

            if (u_ref_i_s - border < 0 || v_ref_i_s - border < 0 || u_ref_i_s + border >= patch.full_img.cols ||
                v_ref_i_s + border >= patch.full_img.rows)
                continue;
            if (u_ref_i_e - border < 0 || v_ref_i_e - border < 0 || u_ref_i_e + border >= patch.full_img.cols ||
                v_ref_i_e + border >= patch.full_img.rows)
                continue;

            *visiblity_it = true;


            Vector2f inc2f;

            size_t N_samples = setupSampling(patch.size, inc2f, sp_edge, ep_edge, length);

            N_samples = 1 + (N_samples-1) * scale;

            // Parameterize 2D segment
            inc2f = inc2f * scale / (N_samples-1);
            Vector2f px_ref = sp_edge * scale;

            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;


            std::vector<Vector2f> tem;
            std::vector<Vector2f> tem1;

            for(unsigned int sample = 0; sample < N_samples; ++sample, px_ref+=inc2f, xyz_ref+=inc3f )
            {
                patch.setPosition( px_ref );
                patch.computeInterpWeights();
                patch.setRoi();

                tem.push_back(px_ref);
                tem1.push_back(ep_edge * scale);

                // evaluate projection jacobian
                Matrix<float,2,6> frame_jac;
                frame_jac = JacobXYZ2Cam(xyz_ref);

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;
                const int stride = patch.stride;
                std::vector<float> cache_ptr_edge;

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                        cache_ptr_edge.push_back(*cache_ptr);

                        float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
                                           -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
                        float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
                                           -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));

                        // cache the jacobian
                        edge_cache_.jacobian.col(cache_idx) =
                                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length * scale);

                    }//end col-sweep in current row
                }//end row-sweep
                cache.seg_ref_patch[i][sample] = cache_ptr_edge;
            }//end segment-sweep (through sampled patches)
            preCompute[i] = tem;
            preComputeEnd[i] = tem1;
        }//end feature-sweep
    }

    void SparseImgAlign::precomputeGaussNewtonParamsLKSegments(Cache &cache)
    {
        // initialize patch parameters (mainly define its geometry)
        Patch patch( patch_size_, ref_frame_->mvImagePyramid_zzw[level_] );

        const float scale = ref_frame_->mvInvScaleFactors[level_];
        const float focal_length = ref_frame_->fx;
        const int border = patch_halfsize_ + 1;

        // TODO: feature_counter is no longer valid because each segment
        //  has a variable number of patches (and total pixels)
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();
        int NLK = ref_frame_->mLKLine.size();

        patch_offset_lkSeg = std::vector<size_t>(NLK,0);
        std::vector<size_t>::iterator offset_it = patch_offset_lkSeg.begin();
        size_t cache_idx = 0;

        for(int i = 0; i < NLK; i++, ++visiblity_it, ++offset_it)
        {
            *offset_it = cache_idx;

            Vector2f sp_l,ep_l;
            float length;
            sp_l<<ref_frame_->mLKLine[i].first[0],ref_frame_->mLKLine[i].first[1];
            ep_l<<ref_frame_->mLKLine[i].second[0],ref_frame_->mLKLine[i].second[1];
            length = (ep_l-sp_l).norm();


            const float u_ref_s = sp_l[0]*scale;
            const float v_ref_s = sp_l[1]*scale;
            const float u_ref_e = ep_l[0]*scale;
            const float v_ref_e = ep_l[1]*scale;
            const int u_ref_i_s = floorf(u_ref_s);
            const int v_ref_i_s = floorf(v_ref_s);
            const int u_ref_i_e = floorf(u_ref_e);
            const int v_ref_i_e = floorf(v_ref_e);

            if (u_ref_i_s - border < 0 || v_ref_i_s - border < 0 || u_ref_i_s + border >= patch.full_img.cols ||
                v_ref_i_s + border >= patch.full_img.rows)
                continue;
            if (u_ref_i_e - border < 0 || v_ref_i_e - border < 0 || u_ref_i_e + border >= patch.full_img.cols ||
                v_ref_i_e + border >= patch.full_img.rows)
                continue;


            std::pair<float, float> depth;
            depth = make_pair(ref_frame_->mLKLine[i].first[2],ref_frame_->mLKLine[i].second[2]);
            if (depth.first <= 0 || depth.second <= 0)
                continue;

            *visiblity_it = true;

            Vector2f inc2f;

            size_t N_samples = setupSampling(patch.size, inc2f, sp_l, ep_l, length);

            N_samples = 1 + (N_samples-1) * scale;

            // Parameterize 2D segment
            inc2f = inc2f * scale / (N_samples-1);
            Vector2f px_ref = sp_l * scale;


            Vector3f p_ref,q_ref;
            p_ref[0] = (sp_l[0] - ref_frame_->cx) * depth.first * ref_frame_->invfx;
            p_ref[1] = (sp_l[1] - ref_frame_->cy) * depth.first * ref_frame_->invfy;
            p_ref[2] = depth.first;
            q_ref[0] = (ep_l[0] - ref_frame_->cx) * depth.second * ref_frame_->invfx;
            q_ref[1] = (ep_l[1] - ref_frame_->cy) * depth.second * ref_frame_->invfy;
            q_ref[2] = depth.second;


            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;

            for(unsigned int sample = 0; sample<N_samples; ++sample, px_ref+=inc2f, xyz_ref+=inc3f )
            {
                patch.setPosition( px_ref );
                patch.computeInterpWeights();
                patch.setRoi();

                // evaluate projection jacobian
                Matrix<float,2,6> frame_jac;
                frame_jac = JacobXYZ2Cam(xyz_ref);

                if (cur_frame_->mnId>111110){
                    cout<<"xyz_ref: "<<xyz_ref[0]<<" "<<xyz_ref[1]<<" "<<xyz_ref[2]<<endl;
                    cout<<"depth:   "<<depth.first<<"  "<<depth.second<<endl;
                    cout<<"frame_jac:   "<<frame_jac<<endl;
                    getchar();
                }

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;
                const int stride = patch.stride;
                std::vector<float> cache_ptr_lkSeg;

                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                        cache_ptr_lkSeg.push_back(*cache_ptr);

                        float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
                                           -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
                        float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
                                           -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));

                        // cache the jacobian
                        lkSeg_cache_.jacobian.col(cache_idx) =
                                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length * scale);

                    }//end col-sweep in current row
                }//end row-sweep
                cache.seg_ref_patch[i][sample] = cache_ptr_lkSeg;
            }//end segment-sweep (through sampled patches)
        }//end feature-sweep
    }

    int SparseImgAlign::solve() {
        x_ = H_.ldlt().solve(Jres_);
        if ((bool) std::isnan((float) x_[0]))
            return 0;
        return 1;
    }

    void SparseImgAlign::update(
            const ModelType &T_curold_from_ref,
            ModelType &T_curnew_from_ref) {
        T_curnew_from_ref = T_curold_from_ref * SE3f::exp(-x_);
    }

    void SparseImgAlign::startIteration() {}

    void SparseImgAlign::finishIteration() {
        if (display_) {
            cv::namedWindow("residuals", CV_WINDOW_AUTOSIZE);
            cv::imshow("residuals", resimg_ * 10);
            cv::waitKey(0);
        }
    }

    void SparseImgAlign::mat2SE3(cv::Mat org,SE3f &dst){
        cv::Mat org_R = org.rowRange(0,3).colRange(0,3);
        cv::Mat org_t = org.rowRange(0,3).col(3);
        Eigen::Matrix3f org_R_Matrix;
        Eigen::Vector3f org_t_Matrix;
        cv::cv2eigen(org_R,org_R_Matrix);
        cv::cv2eigen(org_t,org_t_Matrix);
        SE3f dst_SE3(org_R_Matrix,org_t_Matrix);
        dst = dst_SE3;
    }
} // namespace Planar_SLAM