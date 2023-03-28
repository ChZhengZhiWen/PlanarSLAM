#include "SparseImageAlign.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Feature.h"

namespace Planar_SLAM {
using namespace cv::line_descriptor;

    SparseImgAlign::SparseImgAlign(
            int max_level, int min_level, int n_iter,
            Method method, bool display, bool verbose) :
            display_(display),
            max_level_(max_level),
            min_level_(min_level) {
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
        int max_num_seg_samples = std::ceil( total_length / patch_size_ );

        pt_cache_  = Cache( ref_frame_->N, patch_area_ );
        seg_cache_ = Cache( max_num_seg_samples, patch_area_ );

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

            seg_cache_.visible_fts.resize(seg_cache_.ref_patch.rows,false);

            jacobian_cache_.setZero();
            have_ref_patch_cache_ = false;
//            n_iter_ = iterations[level_];
            n_iter_ = 10;

            segNum=0;

            preCompute_map.clear();
            preCompute3D_map.clear();
            preCompute3DEnd_map.clear();
            preComputeEnd_map.clear();

            seg_cache_.seg_ref_patch.clear();

            optimize(T_cur_from_ref,_chi2);
            cout<<endl;
            if (_chi2 > 700 || n_meas_ < 2500){
                cout<<"_chi2 > x || n_meas_ < y"<<endl;
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

        if(linearize_system)
        {
            // sum the contribution from both points and segments
            H_    = pt_H    + seg_H;
            Jres_ = pt_Jres + seg_Jres;
        }

        float chi2 = pt_chi2 + seg_chi2;

        // compute the weights on the first iteration
        if (compute_weight_scale && iter_ == 0)
            scale_ = scale_estimator_->compute(errors);
        return chi2 / n_meas_;
    }

    /*
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
        // initialize patch parameters (mainly define its geometry)
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

        // set GN parameters to zero prior to accumulate results
        H.setZero();
        Jres.setZero();
        size_t feature_counter = 0; // is used to compute the index of the cached jacobian
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

        for(int i = 0; i < ref_frame_->N; i++, ++feature_counter, ++visiblity_it)
        {
            // check if feature is within image
            if(!*visiblity_it)
                continue;

            // compute pixel location in cur img
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

            // set patch position for current feature
            patch.setPosition(uv_cur_pyr);
            // skip this feature if the patch (with extra pixel for border in derivatives) does not fully lie within the image
            if(!patch.isInFrame(patch.halfsize+1))
                continue;
            // compute the bilinear interpolation weights constant along the patch scan
            patch.computeInterpWeights();
            // set the patch at the corresponding ROI in the image
            patch.setRoi();
            // iterate through all points in the Region Of Interest defined by the patch
            // the pointer points to the data in the original image matrix
            // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
            size_t pixel_counter = 0;
            float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + patch.area*feature_counter;
            uint8_t* img_ptr; // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
            const int stride = patch.stride; // the stride stored in the patch is that necessary to jump between the full matrix rows
            cv::MatIterator_<float> itDisp;
            //display_ default false
            if(linearize_system && display_)
            {
                resPatch.setPosition(uv_cur_pyr);
                resPatch.setRoi();
                itDisp = resPatch.roi.begin<float>();
            }


            if( cur_frame_->mnId>12000 ) {
                cv::Mat img, img1, img2;
                patch.full_img.copyTo(img);
                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img2);

                cv::Point2f curP(uv_cur_pyr[0], uv_cur_pyr[1]);
                cv::Point2f curFeaP(cur_frame_->Camera2Pixel(xyz_ref)[0]*scale, cur_frame_->Camera2Pixel(xyz_ref)[1]*scale);
                cv::Point2f refP(prePointCompute[i][0], prePointCompute[i][1]);


                cout <<"i:"<<i<< " curP:" << curP << " curFeaP:" << curFeaP<<" refP:"<<refP<< endl;
                cout<<"norm:"<<norm((refP-curP))/scale<<endl;
                int curx = floorf(curP.x), cury = floorf(curP.y);
                float curPix = *(patch.full_img.data + cury * stride + curx);
                int prex = floorf(refP.x), prey = floorf(refP.y);
                float prePix = *(ref_frame_->mvImagePyramid_zzw[level_].data + prey * stride + prex);

                cv::circle(img, curP, 2, cv::Scalar(255, 0, 0), 2, -1);
                cv::circle(img1, refP, 2, cv::Scalar(255, 0, 0), 2, -1);
                cv::circle(img2, curFeaP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img2, curFeaP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img3, pre3DP, 2, cv::Scalar(255, 0, 0), 2, -1);


                cv::imshow("curP", img);
                cv::imshow("refP", img1);
                cv::imshow("curFeaP", img2);

                cout << "intensity:" << curPix - prePix << endl;
                if (norm((refP-curP))>20)
                    cout<<"-----------------"<<endl;
                getchar();
            }


            int tem = 0;
            for(int y=0; y<patch.size; ++y) // sweep the path row-wise (most efficient for RowMajor storage)
            {
                // get the pointer to first element in row y of the patch ROI
                // Mat.ptr() acts on the dimension #0 (rows)
                img_ptr = patch.roi.ptr(y);
                for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++pixel_counter)
                {
                    // compute residual
                    const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                    const float res = intensity_cur - (*cache_ptr);
                    const float res2 = res*res;
                    // used to compute scale for robust cost
                    if(compute_weight_scale)
                        errors.push_back(fabsf(res));
tem+=fabsf(res);

                    // robustification
                    float weight = 1.0;

                    if(use_weights_)
                    {
                        if(compute_weight_scale && iter_ != 0)
                        {
                            //weight = 2.0*fabsf(res) / (1.0+res2/scale_pt);
/// zzw                            weight = 1.0 / (1.0+fabsf(res)/scale_pt);
                            //weight = weight_estimator.value(fabsf(res)/scale_pt);
                        }
                        else
                        {
                            //weight = 2.0*fabsf(res) / (1.0+res2);
                            weight = 1.0 / (1.0+fabsf(res));
                            //weight = weight_estimator.value(fabsf(res));
                        }
                    }

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
//cout<<"tem "<<tem<<endl;

        }
    }
*/

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

/*
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
        xyz_ref_compute_map.clear();
        xyz_ref_computeEnd_map.clear();
        xyz_cur_compute_map.clear();
        xyz_cur_computeEnd_map.clear();


        // initialize patch parameters (mainly define its geometry)
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

        // set GN parameters to zero prior to accumulate results
        H.setZero();
        Jres.setZero();
        std::vector<size_t>::iterator offset_it = patch_offset.begin();
        size_t cache_idx = 0; // index of the current pixel as stored in cache
        std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();

        Matrix<float,6,6> H_ls    = Matrix<float, 6, 6>::Zero();
        Matrix<float,6,1> Jres_ls = Matrix<float, 6, 1>::Zero();

        int k =0;

        cv::Mat img__;
        ref_frame_->mvImagePyramid_zzw[level_].copyTo(img__);
        for(int i = 0; i < ref_frame_->NL; i++, ++offset_it, ++visiblity_it)
        {
            // check if feature is within image
            if(!*visiblity_it)
                continue;

            MapLine *ml = ref_frame_->mvpMapLines[i];
            if (!ml || ml->isBad() || ref_frame_->mvbLineOutlier[i]) {
                continue;
            }

            const cv::line_descriptor::KeyLine kl = ref_frame_->mvKeylinesUn[i];
            Vector2f sp_l(kl.startPointX,kl.startPointY),ep_l(kl.endPointX,kl.endPointY);
            float length = (ep_l-sp_l).norm();

            // setup current index in cache according to stored offset values
            cache_idx = *offset_it;

            Vector2f inc2f; // will store the difference vector from start to end points in the segment first
            // later will parameterize the 2D step to sample the segment
            size_t N_samples = setupSampling(patch.size, inc2f, sp_l, ep_l, length);
            // Adjust the number of samples in terms of the current pyramid level
            N_samples = 1 + (N_samples-1) * scale; // for lvl 0 take all, for lvl n downsample by 2^n

inc2f = inc2f * scale / (N_samples-1);
Vector2f px_ref = sp_l * scale;

            // Parameterize 3D segment with start point and discrete 3D increment
            auto tmp = Converter::toSE3Quat(ref_frame_->mTcw);
            SE3f ref_Tcw_SE3 = SE3d(tmp.rotation(), tmp.translation()).cast<float>();
            Vector3f p_ref = ref_Tcw_SE3 * ml->GetWorldPos().head(3).cast<float>();
            Vector3f q_ref = ref_Tcw_SE3 * ml->GetWorldPos().tail(3).cast<float>();

            Vector2f normal_vector_pixel = ep_l - sp_l;
            Vector2f normal_vector_projection = cur_frame_->Camera2Pixel(q_ref) - cur_frame_->Camera2Pixel(p_ref);
            float angle_pixel_projection = normal_vector_pixel[0] * normal_vector_projection[0] + normal_vector_pixel[1] * normal_vector_projection[1];
            angle_pixel_projection /= (normal_vector_pixel.norm()*normal_vector_projection.norm());

            if (angle_pixel_projection < (-0.8) ){
                const Vector3f tem = p_ref;
                p_ref = q_ref;
                q_ref = tem;
            }
//            Vector2f inc2fTem;
//            Vector2f p_cur(cur_frame_->Camera2Pixel(T_cur_from_ref*p_ref));
//            Vector2f q_cur(cur_frame_->Camera2Pixel(T_cur_from_ref*q_ref));
//            length = (q_ref-p_ref).norm();
//            N_samples = setupSampling(patch.size, inc2fTem, p_cur, q_cur, length);
//            N_samples = 1 + (N_samples-1) * scale;
            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;


            // Evaluate over the patch for each point sampled in the segment (including extremes)
            Matrix<float,6,6> H_    = Matrix<float, 6, 6>::Zero();
            Matrix<float,6,1> Jres_ = Matrix<float, 6, 1>::Zero();
            vector<float> ls_res;
            bool good_line = true;


            xyz_ref_compute.clear();
            xyz_ref_computeEnd.clear();
            xyz_cur_compute.clear();
            xyz_cur_computeEnd.clear();
            for(unsigned int sample = 0; sample < N_samples; ++sample, xyz_ref+=inc3f, px_ref+=inc2f,++k )
            {
                // compute pixel location in cur img
                const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);
                const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
                const Vector2f uv_cur_pyr(uv_cur * scale);
                // set patch position for current feature

                patch.setPosition(uv_cur_pyr);
                // skip this feature if the patch (with extra pixel for border in derivatives) does not fully lie within the image
                if(!patch.isInFrame(patch.halfsize+1))
                {
                    cache_idx += patch.size; // Do not lose position of the next patch in cache!
                    good_line = false;
                    sample    = N_samples;
                    continue;
                }
                // compute the bilinear interpolation weights constant along the patch scan
                patch.computeInterpWeights();
                // set the patch at the corresponding ROI in the image
                patch.setRoi();

                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr; // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
                const int stride = patch.stride; // the stride stored in the patch is that necessary to jump between the full matrix rows
                cv::MatIterator_<float> itDisp;
                //display_ default false
                if(linearize_system && display_)
                {
                    resPatch.setPosition(uv_cur_pyr);
                    resPatch.setRoi();
                    itDisp = resPatch.roi.begin<float>();
                }

                xyz_ref_compute.push_back(cur_frame_->Camera2Pixel(xyz_ref)* scale);
                xyz_ref_computeEnd.push_back(cur_frame_->Camera2Pixel(q_ref)* scale);
                xyz_cur_compute.push_back(uv_cur_pyr);
                xyz_cur_computeEnd.push_back(cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)* scale);

//                const float u_cur = px_ref[0];
//                const float v_cur = px_ref[1];
//                //floorf向下取整
//                const int u_cur_i = floorf(u_cur);
//                const int v_cur_i = floorf(v_cur);
//                const float subpix_u_cur = u_cur - u_cur_i;
//                const float subpix_v_cur = v_cur - v_cur_i;
//                const float w_cur_tl = (1.0 - subpix_u_cur) * (1.0 - subpix_v_cur);
//                const float w_cur_tr = subpix_u_cur * (1.0 - subpix_v_cur);
//                const float w_cur_bl = (1.0 - subpix_u_cur) * subpix_v_cur;
//                const float w_cur_br = subpix_u_cur * subpix_v_cur;



int sum = 0;
int ind = 0;
                std::vector<float> cache_ptr_seg = cache.seg_ref_patch[i][sample];
                if( cur_frame_->mnId>15000 && level_ == min_level_) {
                    cv::Mat img, img1, img2,img3;
                    patch.full_img.copyTo(img);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img2);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img3);
                    cv::Point2f curP(uv_cur_pyr[0], uv_cur_pyr[1]);
                    cv::Point2f curPEnd(cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)[0]* scale, cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)[1]* scale);
                    cv::Point2f curFeaP(cur_frame_->Camera2Pixel(xyz_ref)[0]*scale, cur_frame_->Camera2Pixel(xyz_ref)[1]*scale);
                    cv::Point2f curFeaPEnd(cur_frame_->Camera2Pixel(q_ref)[0]*scale, cur_frame_->Camera2Pixel(q_ref)[1]*scale);
                    cv::Point2f preP(preCompute_map[i][sample][0], preCompute_map[i][sample][1]);
                    cv::Point2f prePEnd(preComputeEnd_map[i][sample][0], preComputeEnd_map[i][sample][1]);
                    cv::Point2f pre3DP(cur_frame_->Camera2Pixel(preCompute3D_map[i][sample])[0]*scale, cur_frame_->Camera2Pixel(preCompute3D_map[i][sample])[1]*scale);
                    cv::Point2f pre3DPEnd(cur_frame_->Camera2Pixel(preCompute3DEnd_map[i][sample])[0]*scale, cur_frame_->Camera2Pixel(preCompute3DEnd_map[i][sample])[1]*scale);


                    cout <<"i:"<<i<< " curP:" << curP << " preP:" << preP<<" curFeaP:"<<curFeaP<<" pre3DP:"<<pre3DP<<" pre3DPEnd:"<<pre3DPEnd << endl;
                    cout<<"norm:"<<norm((pre3DP-preP))<<endl;
                    int curx = floorf(curP.x), cury = floorf(curP.y);
                    float curPix = *(patch.full_img.data + cury * stride + curx);
                    int prex = floorf(preP.x), prey = floorf(preP.y);
                    float prePix = *(ref_frame_->mvImagePyramid_zzw[level_].data + prey * stride + prex);

//                    cv::circle(img, curP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img1, preP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img2, curFeaP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img3, pre3DP, 2, cv::Scalar(255, 0, 0), 2, -1);

                    cv::line(img, curP, curPEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                    cv::line(img1, preP, prePEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                    cv::line(img2, curFeaP, curFeaPEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                    cv::line(img3, pre3DP, pre3DPEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);


                    cv::imshow("cur", img);
                    cv::imshow("pre", img1);
                    cv::imshow("curFea", img2);
                    cv::imshow("pre3DP", img3);

                    cout << "intensity:" << curPix - prePix << endl;
                }
                cout<<endl;
                for(int y=0; y<patch.size; ++y) // sweep the path row-wise (most efficient for RowMajor storage)
                {
                    // get the pointer to first element in row y of the patch ROI
                    // Mat.ptr() acts on the dimension #0 (rows)
                    img_ptr = patch.roi.ptr(y);
//                    uint8_t *cur_img_pix = (uint8_t *) img__.data + (v_cur_i + y - patch_halfsize_) * stride +
//                                           (u_cur_i - patch_halfsize_);
//                    uint8_t *pre_img_pix = (uint8_t *) img__.data + (patch.v_ref_i + y - patch_halfsize_) * stride +
//                                           (patch.u_ref_i - patch_halfsize_);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx,++ind)
                    {
                        // compute residual
                        const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
//                        const float intensity_pix =
//                                w_cur_tl * cur_img_pix[0] + w_cur_tr * cur_img_pix[1] + w_cur_bl * cur_img_pix[stride] +
//                                w_cur_br * cur_img_pix[stride + 1];
//                        const float intensity_pre_img =
//                                patch.wTL * pre_img_pix[0] + patch.wTR * pre_img_pix[1] + patch.wBL * pre_img_pix[stride] +
//                                        patch.wBR * pre_img_pix[stride + 1];
                        const float res = intensity_cur - (*cache_ptr);
//                        if (*cache_ptr!=intensity_pix){
//                            cout<<"sdafasdfsaf"<<endl;
//                            getchar();
//                        }
//cout<<intensity_cur<<" "<<*cache_ptr<<" "<<intensity_pre_img<<" | ";
                        sum+=fabsf(res);

//cout<<fabsf(res)<<" ";
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
cout<<"sum: "<<sum<<endl;
                if( cur_frame_->mnId>15000 && sum >500){
                    cout<<"tem: "<<sum<<endl;
                    cv::Mat img,img1,img2;
                    patch.full_img.copyTo(img);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img2);

                    const Vector2f endP(cur_frame_->Camera2Pixel(T_cur_from_ref*q_ref)* scale);
                    cv::Point2f p1 = cv::Point2f(xyz_cur_compute[sample][0], xyz_cur_compute[sample][1]);
                    cv::Point2f q1 = cv::Point2f(xyz_cur_computeEnd[sample][0], xyz_cur_computeEnd[sample][1]);
                    cv::Point2f p2 = cv::Point2f(preCompute_map[i][sample][0],preCompute_map[i][sample][1]);
                    cv::Point2f q2 = cv::Point2f(preComputeEnd_map[i][sample][0],preComputeEnd_map[i][sample][1]);
                    cv::Point2f p3 = cv::Point2f(px_ref[0],px_ref[1]);
                    cv::Point2f q3 = cv::Point2f(ep_l[0]*scale,ep_l[1]*scale);
                    cv::line(img, p1, q1, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
                    cv::line(img1, p2, q2, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
                    cv::line(img2, p3, q3, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
                    cv::imshow("cur",img);
                    cv::imshow("cur_ref_feature",img2);
                    cv::imshow("pre",img1);

                    cout<<"p1:"<<p1<<" q1:"<<q1<<endl;
                    cout<<"p2:"<<p2<<" q2:"<<q2<<endl;
                    cout<<"p3:"<<p3<<" q3:"<<q3<<endl;

                    getchar();
                }
cout<<endl;
if( cur_frame_->mnId>15000 && level_ == min_level_)
    getchar();


            }//end segment-sweep

            xyz_ref_compute_map.insert(std::pair<int,std::vector<Vector2f>>(i,xyz_ref_compute));
            xyz_ref_computeEnd_map.insert(std::pair<int,std::vector<Vector2f>>(i,xyz_ref_computeEnd));
            xyz_cur_compute_map.insert(std::pair<int,std::vector<Vector2f>>(i,xyz_cur_compute));
            xyz_cur_computeEnd_map.insert(std::pair<int,std::vector<Vector2f>>(i,xyz_cur_computeEnd));

            float res_ = 0.0, res2_;
            for(vector<float>::iterator it = ls_res.begin(); it != ls_res.end(); ++it){
                //res_ += pow(*it,2);
                res_ += fabsf(*it);
                //res_ += pow(*it,2);
            }
            //res_ = sqrt(res_)/double(N_samples) ;
            res_ = res_ / double(N_samples);
//            cout<<"res_:"<<res_<<endl;

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

        if (cur_frame_->mnId>5000){
            cout<<"xyz_ref_computeSize:"<<xyz_ref_compute.size()<<"  xyz_cur_computeSize:"<<xyz_cur_compute.size()
                <<"  preComputeSize:"<<preCompute.size()<<endl;
            for (int i = 0; i<xyz_cur_compute.size();i++) {
//                if (i%5 == 0)
//                    getchar();
                cv::Mat img,img1,img2;
                patch.full_img.copyTo(img);
                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img2);

                cv::Point p1 = cv::Point(xyz_cur_compute[i][0], xyz_cur_compute[i][1]);
                cv::Point q1 = cv::Point(xyz_cur_computeEnd[i][0], xyz_cur_computeEnd[i][1]);

                cv::Point p2 = cv::Point(xyz_ref_compute[i][0], xyz_ref_compute[i][1]);
                cv::Point q2 = cv::Point(xyz_ref_computeEnd[i][0], xyz_ref_computeEnd[i][1]);

                cv::Point p3 = cv::Point(preCompute[i][0], preCompute[i][1]);
                cv::Point q3 = cv::Point(preComputeEnd[i][0], preComputeEnd[i][1]);

                cv::line(img, p1, q1, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                cv::line(img1, p2, q2, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                cv::line(img2, p3, q3, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                cv::imshow("p1",img);
                cv::imshow("p2",img1);
                cv::imshow("p3",img2);
                cout<<"i:"<<i<<endl;
                cout<<"xyz_ref_compute    "<<xyz_ref_compute[i][0]<<","<<xyz_ref_compute[i][1]<<endl;
                cout<<"preCompute         "<<preCompute[i][0]<<","<<preCompute[i][1]<<endl;
                cout<<"xyz_cur_compute    "<<xyz_cur_compute[i][0]<<","<<xyz_cur_compute[i][1]<<endl<<endl;
                getchar();
            }
            getchar();
        }

    }
*/

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
int nl=0,i1=0;
        for(int i = 0; i < ref_frame_->NL; i++, ++offset_it, ++visiblity_it)
        {
            if(!*visiblity_it)
                continue;

//            MapLine *ml = ref_frame_->mvpMapLines[i];
//            if (!ml || ml->isBad() || ref_frame_->mvbLineOutlier[i]) {
//                continue;
//            }
//
            std::pair<float, float> depth = ref_frame_->mvDepthLine_zzw[i];
//            if (depth.first < 0 || depth.second < 0)
//                continue;

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
nl++;
            for(unsigned int sample = 0; sample < N_samples; ++sample, xyz_ref+=inc3f )
            {
                const Vector3f xyz_cur(T_cur_from_ref * xyz_ref);
                const Vector2f uv_cur(cur_frame_->Camera2Pixel(xyz_cur));
                const Vector2f uv_cur_pyr(uv_cur * scale);

                if( cur_frame_->mnId>40000 && level_ == 3) {
                    cv::Mat img, img1;
                    patch.full_img.copyTo(img);
                    ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
                    cv::Point2f curP(uv_cur_pyr[0], uv_cur_pyr[1]);
                    cv::Point2f curPEnd(cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)[0]* scale, cur_frame_->Camera2Pixel(T_cur_from_ref * q_ref)[1]* scale);

                    cv::Point2f preP(preCompute_map[i][sample][0], preCompute_map[i][sample][1]);
                    cv::Point2f prePEnd(preComputeEnd_map[i][sample][0], preComputeEnd_map[i][sample][1]);
//                    cv::Point2f pre3DP(cur_frame_->Camera2Pixel(preCompute3D_map[i][sample])[0]*scale, cur_frame_->Camera2Pixel(preCompute3D_map[i][sample])[1]*scale);
//                    cv::Point2f pre3DPEnd(cur_frame_->Camera2Pixel(preCompute3DEnd_map[i][sample])[0]*scale, cur_frame_->Camera2Pixel(preCompute3DEnd_map[i][sample])[1]*scale);
                    cv::Point3f pre3DP(preCompute3D_map[i][sample][0], preCompute3D_map[i][sample][1],preCompute3D_map[i][sample][2]);


                    cout <<"i:"<<i<< " curP:" << curP << " preP:" << preP<<" pre3DP:"<<pre3DP<< endl;
                    cout <<"xyz_cur ["<<xyz_cur[0]<<" "<<xyz_cur[1]<<" "<<xyz_cur[2]<<"]"<< endl;
                    cout <<"uv_cur ["<<uv_cur[0]<<" "<<uv_cur[1]<<"]"<< endl;
                    cout <<"sp_l*s ["<<sp_l[0]*scale<<" "<<sp_l[1]*scale<<"]"<< endl;
                    cout <<"ep_l*s ["<<ep_l[0]*scale<<" "<<ep_l[1]*scale<<"]"<< endl;
                    cout <<"depth "<<depth.first<<" "<<depth.second<< endl;

//                    cout <<"T_cur_from_ref  :"<<T_cur_from_ref.matrix()<< endl;
                    cout<<"norm:"<<norm((preP-curP))<<endl<<endl;

//                    int curx = floorf(curP.x), cury = floorf(curP.y);
//                    float curPix = *(patch.full_img.data + cury * stride + curx);
//                    int prex = floorf(preP.x), prey = floorf(preP.y);
//                    float prePix = *(ref_frame_->mvImagePyramid_zzw[level_].data + prey * stride + prex);

//                    cv::circle(img, curP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img1, preP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img2, curFeaP, 2, cv::Scalar(255, 0, 0), 2, -1);
//                    cv::circle(img3, pre3DP, 2, cv::Scalar(255, 0, 0), 2, -1);

                    cv::line(img, curP, curPEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
                    cv::line(img1, preP, prePEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_8);


                    cv::imshow("cur", img);
                    cv::imshow("pre", img1);
//                    cout << "intensity:" << curPix - prePix << endl;
                    if (norm((preP-curP)) > 50)
                    getchar();
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


int sum = 0;
                i1++;
                for(int y=0; y<patch.size; ++y)
                {
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        const float intensity_cur = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];

                        const float res = intensity_cur - (*cache_ptr);
sum+=fabsf(res);
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
//cout<<"sum:"<<sum<<endl;
//cout<<endl;
//getchar();
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

    void SparseImgAlign::precomputeReferencePatches_zzw()
    {
        precomputeGaussNewtonParamsPoints(pt_cache_);
        precomputeGaussNewtonParamsSegments(seg_cache_);
        // set flag to true to avoid repeating unnecessary computations in the following iterations
        have_ref_patch_cache_ = true;
    }

    /*
    void SparseImgAlign::precomputeGaussNewtonParamsPoints(Cache &cache)
    {

        // initialize patch parameters (mainly define its geometry)
        Patch patch( patch_size_, ref_frame_->mvImagePyramid_zzw[level_] );
//        const float scale = 1.0f/(1<<level_);
        const float scale = ref_frame_->mvInvScaleFactors[level_];
        const float focal_length = ref_frame_->fx;

        prePointCompute.clear();


        {
            size_t feature_counter = 0;
            std::vector<bool>::iterator visiblity_it = cache.visible_fts.begin();
            for(int i = 0; i < ref_frame_->N; i++, ++feature_counter, ++visiblity_it)
            {
                // if point is not valid (empty or null) skip this feature
                MapPoint *mp = ref_frame_->mvpMapPoints[i];
                if (mp == nullptr || mp->isBad() || ref_frame_->mvbOutlier[i] == true)
                    continue;

                // set patch position for current feature
                const cv::KeyPoint &kp = ref_frame_->mvKeys[i];
                Vector2f tem(kp.pt.x,kp.pt.y);
                patch.setPosition(tem*scale);

                // skip this feature if the patch (with extra pixel for border in derivatives) does not fully lie within the image
                if(!patch.isInFrame(patch.halfsize+1))
                    continue;


                // compute the bilinear interpolation weights constant along the patch scan
                patch.computeInterpWeights();
                // set the patch at the corresponding ROI in the image
                patch.setRoi();


                // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
                auto tmp = Converter::toSE3Quat(ref_frame_->mTcw);
                SE3f ref_Tcw_SE3 = SE3d(tmp.rotation(), tmp.translation()).cast<float>();
                Vector3f WorldPos;
                cv::cv2eigen(mp->GetWorldPos(),WorldPos);
                const Vector3f xyz_ref = ref_Tcw_SE3 * WorldPos;
//                const Vector3f xyz_ref = ref_frame_->mTcw * mp->GetWorldPos();

                if ((ref_frame_->Camera2Pixel(xyz_ref) - tem).norm()*scale > normTh*scale)
                    continue;


                // flag the feature as valid/visible
                *visiblity_it = true;



                // evaluate projection jacobian
                Matrix<float,2,6> frame_jac;
                frame_jac = JacobXYZ2Cam(xyz_ref);

                // iterate through all points in the Region Of Interest defined by the patch
                // the pointer points to the data in the original image matrix
                // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
                size_t pixel_counter = 0;
                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + patch.area*feature_counter;
                uint8_t* img_ptr;                 // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
                const int stride = patch.stride;  // the stride stored in the patch is that necessary to jump between the full matrix rows

                prePointCompute[i]=tem*scale;


                for(int y=0; y<patch.size; ++y)   // sweep the path row-wise (most efficient for RowMajor storage)
                {
                    // get the pointer to first element in row y of the patch ROI
                    // Mat.ptr() acts on the dimension #0 (rows)
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++pixel_counter)
                    {
                        // precompute interpolated reference patch color
                        *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                        // we use the inverse compositional: thereby we can take the gradient always at the same position
                        // get gradient of warped image (~gradient at warped position)
                        float dx = 0.5f * ((patch.wTL*img_ptr[1] + patch.wTR*img_ptr[2] + patch.wBL*img_ptr[stride+1] + patch.wBR*img_ptr[stride+2])
                                           -(patch.wTL*img_ptr[-1] + patch.wTR*img_ptr[0] + patch.wBL*img_ptr[stride-1] + patch.wBR*img_ptr[stride]));
                        float dy = 0.5f * ((patch.wTL*img_ptr[stride] + patch.wTR*img_ptr[1+stride] + patch.wBL*img_ptr[stride*2] + patch.wBR*img_ptr[stride*2+1])
                                           -(patch.wTL*img_ptr[-stride] + patch.wTR*img_ptr[1-stride] + patch.wBL*img_ptr[0] + patch.wBR*img_ptr[1]));


                        // cache the jacobian
                        //pt_cache_.jacobian.col = [1,2]*[2,6]
                        // focal_length / (1<<level_) 图像缩放内参矩阵随之缩放
                        pt_cache_.jacobian.col(feature_counter*patch.area + pixel_counter) =
                                (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length * scale);

                    }
                }
            }
        }

    }
*/

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


/*
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
        patch_offset = std::vector<size_t>(ref_frame_->NL,0); // vector of offsets in cache for each patch
        std::vector<size_t>::iterator offset_it = patch_offset.begin();
        size_t cache_idx = 0; // index of the current pixel as stored in cache
        for(int i = 0; i < ref_frame_->NL; i++, ++visiblity_it, ++offset_it)
        {
            // set cache index to current feature offset
            *offset_it = cache_idx;

            // if line segment is not valid (empty or null) skip this feature
            MapLine *ml = ref_frame_->mvpMapLines[i];
            if (!ml || ml->isBad() || ref_frame_->mvbLineOutlier[i]) {
                continue;
            }

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

//        if (cur_frame_->mnId>10 && ml){
//
//            auto tmp = Converter::toSE3Quat(ref_frame_->mTcw);
//            SE3f ref_Tcw_SE3 = SE3d(tmp.rotation(), tmp.translation()).cast<float>();
//            Vector3f p_ref = ref_Tcw_SE3 * ml->GetWorldPos().head(3).cast<float>();
//            Vector3f q_ref = ref_Tcw_SE3 * ml->GetWorldPos().tail(3).cast<float>();
//                cv::Mat img,img1;
//                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img);
//                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
//
//                cv::Point2f p1(cur_frame_->Camera2Pixel(p_ref)[0]*scale,cur_frame_->Camera2Pixel(p_ref)[1]*scale);
//                cv::Point2f q1(cur_frame_->Camera2Pixel(q_ref)[0]*scale,cur_frame_->Camera2Pixel(q_ref)[1]*scale);
//                cv::Point2f p2(sp_l[0]*scale,sp_l[1]*scale);
//                cv::Point2f q2(ep_l[0]*scale,ep_l[1]*scale);
//                cv::line(img, p1, q1, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
//                cv::circle(img, p1, 2, cv::Scalar(255, 0, 0), 2, -1);
//                cv::line(img1, p2, q2, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
//                cv::circle(img1, p2, 2, cv::Scalar(255, 0, 0), 2, -1);
//
//                cv::imshow("1",img);
//                cv::imshow("2",img1);
//
//                getchar();
//        }


            // Parameterize 3D segment with start point and discrete 3D increment
            auto tmp = Converter::toSE3Quat(ref_frame_->mTcw);
            SE3f ref_Tcw_SE3 = SE3d(tmp.rotation(), tmp.translation()).cast<float>();
            Vector3f p_ref = ref_Tcw_SE3 * ml->GetWorldPos().head(3).cast<float>();
            Vector3f q_ref = ref_Tcw_SE3 * ml->GetWorldPos().tail(3).cast<float>();

            Vector2f normal_vector_pixel = ep_l - sp_l;
            Vector2f normal_vector_projection = cur_frame_->Camera2Pixel(q_ref) - cur_frame_->Camera2Pixel(p_ref);
            float angle_pixel_projection = normal_vector_pixel[0] * normal_vector_projection[0] + normal_vector_pixel[1] * normal_vector_projection[1];
            angle_pixel_projection /= (normal_vector_pixel.norm()*normal_vector_projection.norm());

            if(angle_pixel_projection < (-0.8)){
                const Vector3f tem = p_ref;
                p_ref = q_ref;
                q_ref = tem;
            }




//            if (cur_frame_->mnId>100 && (cur_frame_->Camera2Pixel(p_ref)-sp_l).norm()*scale>normTh*scale){
//                cv::Mat img,img1;
//                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img);
//                ref_frame_->mvImagePyramid_zzw[level_].copyTo(img1);
//
//                cv::Point2f p1(cur_frame_->Camera2Pixel(p_ref)[0]*scale,cur_frame_->Camera2Pixel(p_ref)[1]*scale);
//                cv::Point2f q1(cur_frame_->Camera2Pixel(q_ref)[0]*scale,cur_frame_->Camera2Pixel(q_ref)[1]*scale);
//                cv::Point2f p2(sp_l[0]*scale,sp_l[1]*scale);
//                cv::Point2f q2(ep_l[0]*scale,ep_l[1]*scale);
//                cv::line(img, p1, q1, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
//                cv::circle(img, p1, 2, cv::Scalar(255, 0, 0), 2, -1);
//                cv::line(img1, p2, q2, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
//                cv::circle(img1, p2, 2, cv::Scalar(255, 0, 0), 2, -1);
//
//                cv::imshow("1",img);
//                cv::imshow("2",img1);
//                cout<<"p1:"<<p1<<" p2:"<<p2<<endl;
//                cout<<"normOrg:"<<(cur_frame_->Camera2Pixel(p_ref)-sp_l).norm()<<endl;
//                cout<<"normOrg1:"<<(cur_frame_->Camera2Pixel(p_ref)-sp_l).norm()*scale<<endl;
//                getchar();
//                continue;
//            }



            // flag the feature as valid/visible
            *visiblity_it = true;

            // Compute the number of samples and total increment
            Vector2f inc2f; // will store the difference vector from start to end points in the segment first
            // later will parameterize the 2D step to sample the segment
            size_t N_samples = setupSampling(patch.size, inc2f, sp_l, ep_l, length);
            // Adjust the number of samples in terms of the current pyramid level
            N_samples = 1 + (N_samples-1) * scale; // for lvl 0 take all, for lvl n downsample by 2^n

            // Parameterize 2D segment
            inc2f = inc2f * scale / (N_samples-1); // -1 to get nr of intervals
            Vector2f px_ref = sp_l * scale; // 2D point in the image segment (to update in the loop), initialize at start 2D point


            Vector3f inc3f = (q_ref-p_ref) / (N_samples-1);
            Vector3f xyz_ref = p_ref;

            preCompute.clear();
            preCompute3D.clear();
            preCompute3DEnd.clear();
            preComputeEnd.clear();
            // Evaluate over the patch for each point sampled in the segment (including extremes)
            for(unsigned int sample = 0; sample<N_samples; ++sample, px_ref+=inc2f, xyz_ref+=inc3f )
            {
                preCompute.push_back(px_ref);
                preCompute3D.push_back(xyz_ref);
                preCompute3DEnd.push_back(q_ref);
                preComputeEnd.push_back(ep_l * scale);

                // set patch position for current point in the segment
                patch.setPosition( px_ref );
                // compute the bilinear interpolation weights constant along the patch scan
                patch.computeInterpWeights();
                // set the patch at the corresponding ROI in the image
                patch.setRoi();

                // evaluate projection jacobian
                Matrix<float,2,6> frame_jac;
                frame_jac = JacobXYZ2Cam(xyz_ref);

                // iterate through all points in the Region Of Interest defined by the patch
                // the pointer points to the data in the original image matrix
                // (this is efficient C-like row-wise scanning of patch, see OpenCV tutorial "How to scan images")
                float* cache_ptr = reinterpret_cast<float*>(cache.ref_patch.data) + cache_idx;
                uint8_t* img_ptr;                 // pointer that will point to memory locations of the ROI (same memory as for the original full ref_img)
                const int stride = patch.stride;  // the stride stored in the patch is that necessary to jump between the full matrix rows
                std::vector<float> cache_ptr_seg;
                for(int y=0; y<patch.size; ++y)   // sweep the path row-wise (most efficient for RowMajor storage)
                {
                    // get the pointer to first element in row y of the patch ROI
                    // Mat.ptr() acts on the dimension #0 (rows)
                    img_ptr = patch.roi.ptr(y);
                    for(int x=0; x<patch.size; ++x, ++img_ptr, ++cache_ptr, ++cache_idx)
                    {
                        // precompute interpolated reference patch color
                        *cache_ptr = patch.wTL*img_ptr[0] + patch.wTR*img_ptr[1] + patch.wBL*img_ptr[stride] + patch.wBR*img_ptr[stride+1];
                        cache_ptr_seg.push_back(*cache_ptr);
                        // we use the inverse compositional: thereby we can take the gradient always at the same position
                        // get gradient of warped image (~gradient at warped position)
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
            preCompute_map.insert(std::pair<int,std::vector<Vector2f>>(i,preCompute));
            preCompute3D_map.insert(std::pair<int,std::vector<Vector3f>>(i,preCompute3D));
            preCompute3DEnd_map.insert(std::pair<int,std::vector<Vector3f>>(i,preCompute3DEnd));
            preComputeEnd_map.insert(std::pair<int,std::vector<Vector2f>>(i,preComputeEnd));

        }//end feature-sweep
    }
*/

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
int nl=0,nl1=0;
        for(int i = 0; i < ref_frame_->NL; i++, ++visiblity_it, ++offset_it)
        {
            *offset_it = cache_idx;

            MapLine *ml = ref_frame_->mvpMapLines[i];
            if (!ml || ml->isBad() || ref_frame_->mvbLineOutlier[i]) {
                continue;
            }

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

            preCompute.clear();
            preComputeEnd.clear();
            preCompute3D.clear();
nl++;
            for(unsigned int sample = 0; sample<N_samples; ++sample, px_ref+=inc2f, xyz_ref+=inc3f )
            {
                preCompute.push_back(px_ref);
                preComputeEnd.push_back(ep_l*scale);
                preCompute3D.push_back(xyz_ref);
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
                std::vector<float> cache_ptr_seg;
nl1++;
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
            preCompute_map.insert(std::pair<int,std::vector<Vector2f>>(i,preCompute));
            preComputeEnd_map.insert(std::pair<int,std::vector<Vector2f>>(i,preComputeEnd));
            preCompute3D_map.insert(std::pair<int,std::vector<Vector3f>>(i,preCompute3D));
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