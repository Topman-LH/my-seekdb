/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase
{
namespace sql
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc,
                        T_FUN_SYS_AI_EMBED,
                        N_AI_EMBED,
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL,
                        NOT_ROW_DIMENSION)
{
}

ObExprAIEmbed::~ObExprAIEmbed()
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::build_dimension_config(ObIAllocator &allocator,
                                          const ObDatum *arg_dim,
                                          ObJsonObject *&config)
{
  INIT_SUCC(ret);
  config = nullptr;
  if (OB_ISNULL(arg_dim) || arg_dim->is_null()) {
    // no dimension config
  } else {
    int64_t dim = arg_dim->get_int();
    ObJsonInt *dim_json = nullptr;
    if (dim <= 0) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, dim, dim_json))) {
    } else if (OB_FAIL(config->add("dimensions", dim_json))) {
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res)
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (OB_ISNULL(arg_model_id) || OB_ISNULL(arg_content)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (arg_model_id->is_null() || arg_content->is_null()) {
    // NULL input propagates as NULL (table batch rows with NULL content)
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);

    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id;
    ObString content;
    ObJsonObject *config = nullptr;
    // TEXT/LONGTEXT inputs may carry a LOB header in the datum; read through
    // ObTextStringHelper so the real payload is sent to the AI service.
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(
            ctx.exec_ctx_, temp_allocator, *arg_model_id, expr.args_[MODEL_IDX]->datum_meta_,
            expr.args_[MODEL_IDX]->obj_meta_.has_lob_header(), model_id))) {
      LOG_WARN("fail to read model id string", K(ret));
    } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
            ctx.exec_ctx_, temp_allocator, *arg_content, expr.args_[CONTENT_IDX]->datum_meta_,
            expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), content))) {
      LOG_WARN("fail to read content string", K(ret));
    } else if (model_id.empty() || content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    } else if (OB_FAIL(build_dimension_config(temp_allocator, arg_dim, config))) {
      res.set_null();
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                       const ObBitVector &skip, const int64_t batch_size)
{
  INIT_SUCC(ret);
  ObDatum *res_datums = expr.locate_batch_datums(ctx);
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);

  if (OB_FAIL(expr.args_[MODEL_IDX]->eval_batch(ctx, skip, batch_size))) {
  } else if (OB_FAIL(expr.args_[CONTENT_IDX]->eval_batch(ctx, skip, batch_size))) {
  } else if (expr.arg_cnt_ == 3
             && OB_FAIL(expr.args_[DIM_IDX]->eval_batch(ctx, skip, batch_size))) {
  } else {
    ObDatumVector model_datums = expr.args_[MODEL_IDX]->locate_expr_datumvector(ctx);
    ObDatumVector content_datums = expr.args_[CONTENT_IDX]->locate_expr_datumvector(ctx);
    ObDatumVector dim_datums;
    if (expr.arg_cnt_ == 3) {
      dim_datums = expr.args_[DIM_IDX]->locate_expr_datumvector(ctx);
    }

    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObEvalCtx::BatchInfoScopeGuard batch_info_guard(ctx);
    batch_info_guard.set_batch_size(batch_size);

    ObArray<ObString> contents;
    ObArray<int64_t> row_indexes;
    ObString model_id;
    bool model_id_set = false;
    ObJsonObject *config = nullptr;
    bool config_built = false;

    for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; ++i) {
      if (skip.at(i) || eval_flags.at(i)) {
        continue;
      }
      batch_info_guard.set_batch_idx(i);
      ObDatum *model_datum = model_datums.at(i);
      ObDatum *content_datum = content_datums.at(i);
      ObDatum *dim_datum = (expr.arg_cnt_ == 3) ? dim_datums.at(i) : nullptr;
      if (OB_ISNULL(model_datum) || OB_ISNULL(content_datum)
          || model_datum->is_null() || content_datum->is_null()) {
        // NULL content must not enter the request and must keep result aligned
        res_datums[i].set_null();
        eval_flags.set(i);
        continue;
      }

      ObString cur_model;
      ObString cur_content;
      // TEXT/LONGTEXT column datums may carry a LOB header; read the real
      // payload via ObTextStringHelper so batch requests stay valid.
      if (OB_FAIL(ObTextStringHelper::read_real_string_data(
              ctx.exec_ctx_, temp_allocator, *model_datum, expr.args_[MODEL_IDX]->datum_meta_,
              expr.args_[MODEL_IDX]->obj_meta_.has_lob_header(), cur_model))) {
        LOG_WARN("fail to read model id string", K(ret), K(i));
      } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
              ctx.exec_ctx_, temp_allocator, *content_datum, expr.args_[CONTENT_IDX]->datum_meta_,
              expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), cur_content))) {
        LOG_WARN("fail to read content string", K(ret), K(i));
      } else if (cur_model.empty() || cur_content.empty()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
        res_datums[i].set_null();
        eval_flags.set(i);
        continue;
      }

      ObString model_copy;
      ObString content_copy;
      if (OB_FAIL(ob_write_string(temp_allocator, cur_model, model_copy))) {
      } else if (OB_FAIL(ob_write_string(temp_allocator, cur_content, content_copy))) {
      } else if (!model_id_set) {
        model_id = model_copy;
        model_id_set = true;
      } else if (0 != model_id.compare(model_copy)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("different model id in one ai_embed batch", K(ret), K(model_id), K(model_copy));
      }

      if (OB_FAIL(ret)) {
      } else if (!config_built) {
        if (OB_FAIL(build_dimension_config(temp_allocator, dim_datum, config))) {
        } else {
          config_built = true;
        }
      }

      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(contents.push_back(content_copy))) {
      } else if (OB_FAIL(row_indexes.push_back(i))) {
      }
    }

    if (OB_SUCC(ret) && contents.count() > 0) {
      ObAIFuncExprInfo *info = nullptr;
      share::ObAiModelEndpointInfo resolved_endpoint;
      const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
      query::ObIAiEndpointResolver *endpoint_resolver =
          ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
      ObArray<ObString> results;
      if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
      } else if (OB_ISNULL(endpoint_resolver)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                     model_id, temp_allocator, resolved_endpoint))) {
      } else {
        ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
        if (OB_FAIL(model.call_dense_embedding_vector_v2(contents, config, results))) {
        } else if (results.count() != contents.count()) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("ai_embed batch result count mismatch",
                   K(ret), K(results.count()), K(contents.count()));
        } else {
          for (int64_t j = 0; OB_SUCC(ret) && j < row_indexes.count(); ++j) {
            const int64_t row_idx = row_indexes.at(j);
            batch_info_guard.set_batch_idx(row_idx);
            ObString result = results.at(j);
            if (OB_FAIL(ObAIFuncUtils::set_string_result(
                    expr, ctx, res_datums[row_idx], result))) {
            } else {
              eval_flags.set(row_idx);
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx,
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  UNUSED(expr_cg_ctx);
  UNUSED(raw_expr);
  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
