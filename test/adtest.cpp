#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "adproblem.hpp"

TDS_AD_TEST(DIFF_CERES);
TDS_AD_TEST(DIFF_STAN_REVERSE);
TDS_AD_TEST(DIFF_STAN_FORWARD);
TDS_AD_TEST(DIFF_CPPAD_AUTO);
