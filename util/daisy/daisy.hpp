/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2009
 * Engin Tola
 * web : http://www.engintola.com
 * email : engin.tola+libdaisy@gmail.com
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/*
 "DAISY: An Efficient Dense Descriptor Applied to Wide Baseline Stereo"
 by Engin Tola, Vincent Lepetit and Pascal Fua. IEEE Transactions on
 Pattern Analysis and achine Intelligence, 31 Mar. 2009.
 IEEE computer Society Digital Library. IEEE Computer Society,
 http:doi.ieeecomputersociety.org/10.1109/TPAMI.2009.77

 "A fast local descriptor for dense matching" by Engin Tola, Vincent
 Lepetit, and Pascal Fua. Intl. Conf. on Computer Vision and Pattern
 Recognition, Alaska, USA, June 2008

 OpenCV port by: Cristian Balint <cristian dot balint at gmail dot com>
 */

#include "opencv2/features2d.hpp"


namespace cv
{
namespace xfeatures2d
{

    
    
/** @brief Class implementing DAISY descriptor, described in @cite Tola10

@param radius radius of the descriptor at the initial scale
@param q_radius amount of radial range division quantity
@param q_theta amount of angular range division quantity
@param q_hist amount of gradient orientations range division quantity
@param norm choose descriptors normalization type, where
DAISY::NRM_NONE will not do any normalization (default),
DAISY::NRM_PARTIAL mean that histograms are normalized independently for L2 norm equal to 1.0,
DAISY::NRM_FULL mean that descriptors are normalized for L2 norm equal to 1.0,
DAISY::NRM_SIFT mean that descriptors are normalized for L2 norm equal to 1.0 but no individual one is bigger than 0.154 as in SIFT
@param H optional 3x3 homography matrix used to warp the grid of daisy but sampling keypoints remains unwarped on image
@param interpolation switch to disable interpolation for speed improvement at minor quality loss
@param use_orientation sample patterns using keypoints orientation, disabled by default.

 */
class CV_EXPORTS DAISY : public DescriptorExtractor
{
public:
    enum
    {
        NRM_NONE = 100, NRM_PARTIAL = 101, NRM_FULL = 102, NRM_SIFT = 103,
    };
    static Ptr<DAISY> create( float radius = 15, int q_radius = 3, int q_theta = 8,
                int q_hist = 8, int norm = DAISY::NRM_NONE, InputArray H = noArray(),
                bool interpolation = true, bool use_orientation = false );

    /** @overload
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) = 0;

    /** @overload
     * @param image image to extract descriptors
     * @param roi region of interest within image
     * @param descriptors resulted descriptors array for roi image pixels
     */
    virtual void compute( InputArray image, Rect roi, OutputArray descriptors ) = 0;

    /**@overload
     * @param image image to extract descriptors
     * @param descriptors resulted descriptors array for all image pixels
     */
    virtual void compute( InputArray image, OutputArray descriptors ) = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void get_descriptor( double y, double x, int orientation, float* descriptor ) const = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param H homography matrix for warped grid
     * @param descriptor supplied array for descriptor storage
     */
    virtual bool get_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param descriptor supplied array for descriptor storage
     */
    virtual void get_unnormalized_descriptor( double y, double x, int orientation, float* descriptor ) const = 0;

    /**
     * @param y position y on image
     * @param x position x on image
     * @param orientation orientation on image (0->360)
     * @param H homography matrix for warped grid
     * @param descriptor supplied array for descriptor storage
     */
    virtual bool get_unnormalized_descriptor( double y, double x, int orientation, double* H, float* descriptor ) const = 0;

};
    

} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV
