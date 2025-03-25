/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 *
 * file_output.cpp - Write output to file.
 */

#include "file_output.hpp"
#include <dirent.h>
#include <filesystem>
#include <stdio.h>
#include <sys/types.h>
// #include <mpi.h>

FileOutput::FileOutput(VideoOptions const *options) : Output(options), fp_(nullptr), count_(0), file_start_time_ms_(0)
{
}

FileOutput::~FileOutput()
{
	closeFile();
}

void FileOutput::outputBuffer(void *mem, size_t size, int64_t timestamp_us, uint32_t flags, int num)
{
	// We need to open a new file if we're in "segment" mode and our segment is full
	// (though we have to wait for the next I frame), or if we're in "split" mode
	// and recording is being restarted (this is necessarily an I-frame already).
	if (fp_ == nullptr ||
		(options_->segment && (flags & FLAG_KEYFRAME) &&
		 timestamp_us / 1000 - file_start_time_ms_ > options_->segment) ||
		(options_->split && (flags & FLAG_RESTART)))
	{
		closeFile();
		openFile(timestamp_us, num);
	}

	LOG(2, "FileOutput: output buffer " << mem << " size " << size);
	if (fp_ && size)
	{
		if (fwrite(mem, size, 1, fp_) != 1)
			throw std::runtime_error("failed to write output bytes");
		if (options_->flush)
			fflush(fp_);
	}
}

void FileOutput::openFile(int64_t timestamp_us, int num)
{
	if (options_->output == "-")
		fp_ = stdout;
	else if (!options_->output.empty())
	{
		// Generate the next output file name.

		DIR *dp = opendir("./result/");
		int i = 0;
		struct dirent *ep;
		std::string result_dir = "./result/";

		if (dp != NULL)
		{
			while ((ep = readdir(dp)))
				i++;

			(void)closedir(dp);
		}

		int dir_name = i - 3;

		char filename[256];
		int n = 0;
		if (num == 0)
		{
			std::string path_0 = result_dir + std::to_string(dir_name) + '/' + options_->output + "_" +
								 std::to_string(num) + options_->output_format;
			n = snprintf(filename, sizeof(filename), path_0.c_str(), count_);
		}
		else if (num == 1)
		{
			std::string path_1 = result_dir + std::to_string(dir_name) + '/' + options_->output + "_" +
								 std::to_string(num) + options_->output_format;
			n = snprintf(filename, sizeof(filename), path_1.c_str(), count_);
		}

		count_++;
		if (options_->wrap)
			count_ = count_ % options_->wrap;
		if (n < 0)
			throw std::runtime_error("failed to generate filename");

		fp_ = fopen(filename, "w");
		if (!fp_)
			throw std::runtime_error("failed to open output file " + std::string(filename));
		LOG(2, "FileOutput: opened output file " << filename);

		file_start_time_ms_ = timestamp_us / 1000;
	}
}

void FileOutput::closeFile()
{
	if (fp_)
	{
		if (options_->flush)
			fflush(fp_);
		if (fp_ != stdout)
			fclose(fp_);
		fp_ = nullptr;
	}
}
