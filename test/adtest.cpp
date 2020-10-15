#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "adproblem.hpp"

TDS_AD_TEST(DIFF_CERES);
// TDS_AD_TEST(DIFF_STAN_REVERSE); Fails
TDS_AD_TEST(DIFF_STAN_FORWARD);
TDS_AD_TEST(DIFF_CPPAD_AUTO);
TDS_AD_TEST(DIFF_CPPAD_CODEGEN_AUTO);
