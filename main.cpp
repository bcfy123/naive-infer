//
// Created by wjldw on 2026/7/29.
//

#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("NaiveInfer");
    FLAGS_log_dir = "./log";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}